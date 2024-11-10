/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

#include <shards/core/runtime.hpp>
#include <shards/shardwrapper.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <shards/core/shared.hpp>
#include <shards/core/async.hpp>
#include <shards/core/serialization.hpp>
#include <limits>
#include <memory>
#include <pdqsort.h>
#include <random>
#include <sstream>

// Windows fix
#ifdef MAX_PRIORITY
#undef MAX_PRIORITY
#endif
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace shards {
namespace Genetic {

struct Random {
  static double nextDouble() { return _udis(_gen); }
  static int nextInt() { return _uintdis(_gen); }
  static double nextNormal() { return _ndis(_gen); }
  static double nextNormal1() { return _ndis1(_gen); }

  static inline thread_local std::random_device _rd{};
  static inline thread_local std::mt19937 _gen{_rd()};
  static inline thread_local std::uniform_int_distribution<> _uintdis{};
  static inline thread_local std::uniform_real_distribution<> _udis{0.0, 1.0};
  static inline thread_local std::normal_distribution<> _ndis{0.0, 0.1};
  static inline thread_local std::normal_distribution<> _ndis1{0.0, 1.0};
};

struct Mutant;
struct Evolve {
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("This shard outputs a [Type::Float Type::Wire] pair. This pair represents the wire configuration from the "
                   "current generation that scored the best fitness score and the fitness score itself.");
  }

  static SHOptionalString help() {
    return SHCCSTR("This shard uses genetic algorithm to evolve and change the configuration of the wire specified in the Wire "
                   "parameter. The evolved wires are given a score based on the fitness wire specified in the Fitness Wire "
                   "parameter. The shard then outputs the best wire and its fitness score. This shard should be used in "
                   "conjunction with the Mutant shard.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return _outputType; }
  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _baseWire = value;
      break;
    case 1:
      _fitnessWire = value;
      break;
    case 2:
      _popsize = value.payload.intValue;
      break;
    case 3:
      _mutation = value.payload.floatValue;
      break;
    case 4:
      _crossover = value.payload.floatValue;
      break;
    case 5:
      _extinction = value.payload.floatValue;
      break;
    case 6:
      _elitism = value.payload.floatValue;
      break;
    case 7:
      _threads = std::max(int64_t(1), value.payload.intValue);
      break;
    case 8:
      _coros = value.payload.intValue;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _baseWire;
    case 1:
      return _fitnessWire;
    case 2:
      return shards::Var(_popsize);
    case 3:
      return shards::Var(_mutation);
    case 4:
      return shards::Var(_crossover);
    case 5:
      return shards::Var(_extinction);
    case 6:
      return shards::Var(_elitism);
    case 7:
      return shards::Var(_threads);
    case 8:
      return shards::Var(_coros);
    default:
      return shards::Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    // we still want to run a compose on the subject, in order
    // to serialize it properly!
    assert(_baseWire.valueType == SHType::Wire);
    assert(_fitnessWire.valueType == SHType::Wire);

    auto bwire = SHWire::sharedFromRef(_baseWire.payload.wireValue);
    auto fwire = SHWire::sharedFromRef(_fitnessWire.payload.wireValue);

    SHInstanceData vdata{};
    vdata.wire = bwire.get();
    bwire->mesh = data.wire->mesh;
    auto res = composeWire(bwire.get(), vdata);
    arrayFree(res.exposedInfo);
    arrayFree(res.requiredInfo);
    // unset the wire->mesh as it's not the same as the one we run on, that will happen later
    // but warmup will fail if we don't do this
    bwire->mesh.reset();

    vdata.inputType = res.outputType;
    vdata.wire = fwire.get();
    fwire->mesh = data.wire->mesh;
    res = composeWire(fwire.get(), vdata);
    if (res.outputType.basicType != SHType::Float) {
      throw ComposeError("Evolve: fitness wire should output a Float, but instead got " + type2Name(res.outputType.basicType));
    }
    arrayFree(res.exposedInfo);
    arrayFree(res.requiredInfo);
    // unset the wire->mesh as it's not the same as the one we run on, that will happen later
    // but warmup will fail if we don't do this
    fwire->mesh.reset();

    return _outputType;
  }

  struct Writer {
    std::stringstream &_stream;
    Writer(std::stringstream &stream) : _stream(stream) {}
    void operator()(const uint8_t *buf, size_t size) { _stream.write((const char *)buf, size); }
  };

  struct Reader {
    std::stringstream &_stream;
    Reader(std::stringstream &stream) : _stream(stream) {}
    void operator()(uint8_t *buf, size_t size) { _stream.read((char *)buf, size); }
  };

  void warmup(SHContext *context) {
    const auto threads = std::min(_threads, int64_t(std::thread::hardware_concurrency()));
    if (!_exec || _exec->num_workers() != (size_t(threads) + 1)) {
      _exec.reset(new tf::Executor(size_t(threads) + 1));
    }
  }

  void cleanup(SHContext *context) {
    if (_population.size() > 0) {
      tf::Taskflow cleanupFlow;
      cleanupFlow.for_each(_population.begin(), _population.end(), [&](Individual &i) {
        // Free and release wire
        i.mesh->terminate();
        auto wire = SHWire::sharedFromRef(i.wire.payload.wireValue);
        stop(wire.get());
        destroyVar(i.wire);
        auto fitwire = SHWire::sharedFromRef(i.fitnessWire.payload.wireValue);
        stop(fitwire.get());
        destroyVar(i.fitnessWire);
      });
      _exec->run(cleanupFlow).get();
      _exec.reset(nullptr);
      _sortedPopulation.clear();
      _population.clear();
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    return awaitne(
        context,
        [&]() {
          // Init on the first run!
          // We reuse those wires for every era
          // Only the DNA changes
          if (_population.size() == 0) {
            SHLOG_TRACE("Evolve, first run, init");

            Serialization serial;
            std::stringstream wireStream;
            Writer w1(wireStream);
            serial.reset();
            serial.serialize(_baseWire, w1);
            auto wireStr = wireStream.str();

            std::stringstream fitnessStream;
            Writer w2(fitnessStream);
            serial.reset();
            serial.serialize(_fitnessWire, w2);
            auto fitnessStr = fitnessStream.str();

            _population.resize(_popsize);
            _nkills = size_t(double(_popsize) * _extinction);
            _nelites = size_t(double(_popsize) * _elitism);

            tf::Taskflow initFlow;
            initFlow.for_each(_population.begin(), _population.end(), [&](Individual &i) {
              Serialization deserial;
              std::stringstream i1Stream(wireStr);
              Reader r1(i1Stream);
              deserial.reset();
              deserial.deserialize(r1, i.wire);
              auto wire = SHWire::sharedFromRef(i.wire.payload.wireValue);
              gatherMutants(wire.get(), i.mutants);
              resetState(i);

              std::stringstream i2Stream(fitnessStr);
              Reader r2(i2Stream);
              deserial.reset();
              deserial.deserialize(r2, i.fitnessWire);
            });
            _exec->run(initFlow).get();

            size_t idx = 0;
            for (auto &i : _population) {
              i.idx = idx;
              _sortedPopulation.emplace_back(&i);
              idx++;
            }

            _era = 0;
          } else {
            SHLOG_TRACE("Evolve, crossover");

            // do crossover here, populating tasks properly
            tf::Taskflow crossoverFlow;
            _crossingOver.clear();
            int currentIdx = 0;
            for (auto &ind : _sortedPopulation) {
              ind->crossoverTask.reset();
              if (Random::nextDouble() < _crossover) {
                // In this case this individual
                // becomes the child between two other individuals
                // there is a chance also to keep current values
                // so this is effectively tree way crossover
                // Select from high fitness individuals
                const auto parent0Idx = int(std::pow(Random::nextDouble(), 4) * double(_popsize));
                auto parent0 = _sortedPopulation[parent0Idx];

                const auto parent1Idx = int(std::pow(Random::nextDouble(), 4) * double(_popsize));
                auto parent1 = _sortedPopulation[parent1Idx];

                if (currentIdx != parent0Idx && currentIdx != parent1Idx && parent0Idx != parent1Idx &&
                    parent0->parent0Idx != currentIdx && parent0->parent1Idx != currentIdx && parent1->parent0Idx != currentIdx &&
                    parent1->parent1Idx != currentIdx) {
                  ind->crossoverTask =
                      crossoverFlow.emplace([this, ind, parent0, parent1]() { crossover(*ind, *parent0, *parent1); });
#if 0
              ind.get().crossoverTask.name(std::to_string(currentIdx) + " = " +
                                           std::to_string(parent0Idx) + " + " +
                                           std::to_string(parent1Idx));
#endif
                  _crossingOver.emplace_back(ind, parent0, parent1);
                  ind->parent0Idx = parent0Idx;
                  ind->parent1Idx = parent1Idx;
                }
              }
              currentIdx++;
            }

            for (auto [a, b, c] : _crossingOver) {
              auto &atask = a->crossoverTask;
              auto &btask = b->crossoverTask;
              auto &ctask = c->crossoverTask;
              if (!btask.empty())
                btask.precede(atask);
              if (!ctask.empty())
                ctask.precede(atask);
            }
#if 0
        crossoverFlow.dump(std::cout);
#endif
            _exec->run(crossoverFlow).get();

            _era++;

            // hack/fix
            // it is likely possible that the best wire we outputted
            // was used and so warmedup
            auto best = SHWire::sharedFromRef(_sortedPopulation.front()->wire.payload.wireValue);
            best->cleanup(true);
            best->composedHash = Var::Empty;
            best->wireUsers.clear();
          }

#if 1
          // We run wires up to completion
          // From validation to end, every iteration/era
          // We run in such a way to allow coroutines + threads properly
          SHLOG_TRACE("Evolve, schedule wires");
          {
            tf::Taskflow flow;

            flow.for_each(_era == 0 ? _sortedPopulation.begin() : _sortedPopulation.begin() + _nelites, _sortedPopulation.end(),
                          [](auto &i) {
                            // Evaluate our brain wire
                            auto wire = SHWire::sharedFromRef(i->wire.payload.wireValue);
                            i->mesh->schedule(wire);
                          });

            _exec->run(flow).get();
          }
          SHLOG_TRACE("Evolve, run wires");
          {
            tf::Taskflow flow;

            flow.for_each(_era == 0 ? _sortedPopulation.begin() : _sortedPopulation.begin() + _nelites, _sortedPopulation.end(),
                          [](auto &i) {
                            if (!i->mesh->empty())
                              i->mesh->tick();
                          });

            _exec
                ->run_until(flow,
                            [&]() {
                              size_t ended = 0;
                              for (auto &p : _population) {
                                if (p.mesh->empty())
                                  ended++;
                              }
                              return ended == _population.size();
                            })
                .get();
          }
          SHLOG_TRACE("Evolve, schedule fitness");
          {
            tf::Taskflow flow;

            flow.for_each(_era == 0 ? _sortedPopulation.begin() : _sortedPopulation.begin() + _nelites, _sortedPopulation.end(),
                          [](auto &i) {
                            // reset fitness
                            i->fitness = -std::numeric_limits<float>::max();
                            // avoid scheduling if errors
                            if (!i->mesh->errors().empty())
                              return;
                            // compute the fitness
                            TickObserver obs{*i};
                            auto fitwire = SHWire::sharedFromRef(i->fitnessWire.payload.wireValue);
                            auto wire = SHWire::sharedFromRef(i->wire.payload.wireValue);
                            i->mesh->schedule(obs, fitwire, wire->finishedOutput);
                          });

            _exec->run(flow).get();
          }
          SHLOG_TRACE("Evolve, run fitness");
          {
            tf::Taskflow flow;

            flow.for_each(_era == 0 ? _sortedPopulation.begin() : _sortedPopulation.begin() + _nelites, _sortedPopulation.end(),
                          [](auto &i) {
                            if (!i->mesh->empty()) {
                              TickObserver obs{*i};
                              i->mesh->tick(obs);
                            }
                          });

            _exec
                ->run_until(flow,
                            [&]() {
                              size_t ended = 0;
                              for (auto &p : _population) {
                                if (p.mesh->empty())
                                  ended++;
                              }
                              return ended == _population.size();
                            })
                .get();
          }
#else
          // The following is for reference and for full TSAN runs

          // We run wires up to completion
          // From validation to end, every iteration/era
          {
            tf::Taskflow runFlow;
            runFlow.for_each_dynamic(_population.begin(), _population.end(), [&](Individual &i) {
              TickObserver obs{i};

              // Evaluate our brain wire
              auto wire = SHWire::sharedFromRef(i.wire.payload.wireValue);
              i.mesh->schedule(wire);
              while (!mesh.empty()) {
                i.mesh->tick();
              }

              // reset fitness
              i->fitness = -std::numeric_limits<float>::max();
              // avoid scheduling if errors
              if (!i->mesh->errors().empty())
                return;

              // compute the fitness
              auto fitwire = SHWire::sharedFromRef(i.fitnessWire.payload.wireValue);
              i.mesh->schedule(obs, fitwire, wire->finishedOutput);
              while (!i.mesh->empty()) {
                i.mesh->tick(obs);
              }
            });
            _exec->run(runFlow).get();
          }
#endif
          SHLOG_TRACE("Evolve, stopping all wires");
          { // Stop all the population wires
            tf::Taskflow flow;

            flow.for_each(_population.begin(), _population.end(), [](Individual &i) {
              auto wire = SHWire::sharedFromRef(i.wire.payload.wireValue);
              auto fitwire = SHWire::sharedFromRef(i.fitnessWire.payload.wireValue);
              stop(wire.get());
              wire->composedHash = Var::Empty;
              stop(fitwire.get());
              fitwire->composedHash = Var::Empty;
              i.mesh->terminate();
            });

            _exec->run(flow).get();
          }

          SHLOG_TRACE("Evolve, sorting");
          // remove non normal fitness (sort needs this or crashes will happen)
          std::for_each(_sortedPopulation.begin(), _sortedPopulation.end(), [](auto &i) {
            if (!std::isnormal(i->fitness)) {
              i->fitness = -std::numeric_limits<float>::max();
            }
          });
          pdqsort(_sortedPopulation.begin(), _sortedPopulation.end(),
                  [](const auto &a, const auto &b) { return a->fitness > b->fitness; });

          SHLOG_TRACE("Evolve, resetting flags");
          // reset flags
          std::for_each(_sortedPopulation.begin(), _sortedPopulation.end() - _nkills, [](auto &i) {
            i->extinct = false;
            i->parent0Idx = -1;
            i->parent1Idx = -1;
          });
          std::for_each(_sortedPopulation.end() - _nkills, _sortedPopulation.end(), [](auto &i) {
            i->extinct = true;
            i->parent0Idx = -1;
            i->parent1Idx = -1;
          });

          SHLOG_TRACE("Evolve, run mutations");
          // Do mutations at end, yet when contexts are still valid!
          // since we might need them
          {
            tf::Taskflow mutFlow;
            mutFlow.for_each(_sortedPopulation.begin() + _nelites, _sortedPopulation.end(), [&](auto &i) {
              // reset the individual if extinct
              if (i->extinct) {
                resetState(*i);
              }
              mutate(*i);
            });
            _exec->run(mutFlow).get();
          }

          SHLOG_TRACE("Evolve, era done");

          // hack/fix
          // it is likely possible that the best wire we outputted
          // was used and so warmedup
          auto best = SHWire::sharedFromRef(_sortedPopulation.front()->wire.payload.wireValue);
          best->cleanup(true);
          best->composedHash = Var::Empty;
          best->wireUsers.clear();

          _result.clear();
          _result.emplace_back(shards::Var(_sortedPopulation.front()->fitness));
          _result.emplace_back(_sortedPopulation.front()->wire);
          return shards::Var(_result);
        },
        [] {
          // TODO CANCELLATION
        });
  }

private:
  struct MutantInfo {
    MutantInfo(const Mutant &shard) : shard(shard) {}

    std::reference_wrapper<const Mutant> shard;

    // For reset... hmm we copy them due to multi thread
    // ideally they are all the same tho..
    std::vector<std::tuple<int, OwnedVar>> originalParams;
  };

  static inline void gatherMutants(SHWire *wire, std::vector<MutantInfo> &out);

  struct Individual {
    ~Individual() {
      destroyVar(wire);
      destroyVar(fitnessWire);
    }

    size_t idx = 0;

    // Wires are recycled
    SHVar wire{};
    // We need many of them cos we use threads
    SHVar fitnessWire{};
    // The mesh we run on
    std::shared_ptr<SHMesh> mesh{SHMesh::make()};

    // Keep track of mutants and push/pop mutations on wire
    std::vector<MutantInfo> mutants;

    double fitness{-std::numeric_limits<float>::max()};

    bool extinct = false;

    tf::Task crossoverTask;
    int parent0Idx = -1;
    int parent1Idx = -1;
  };

  struct TickObserver {
    Individual &self;

    void before_tick(SHWire *wire) {}
    void before_start(SHWire *wire) {}
    void before_prepare(SHWire *wire) {}

    void before_stop(SHWire *wire) {
      // Collect fitness last result
      auto fitnessVar = wire->finishedOutput;
      if (fitnessVar.valueType == SHType::Float) {
        self.fitness = fitnessVar.payload.floatValue;
      }
    }

    void before_compose(SHWire *wire) {
      // Keep in mind that individuals might not reach wire termination
      // they might "crash", so fitness should be set to minimum before any run
      self.fitness = -std::numeric_limits<float>::max();
    }
  };

  inline void crossover(Individual &child, const Individual &parent0, const Individual &parent1);
  inline void mutate(Individual &individual);
  inline void resetState(Individual &individual);

  static inline Parameters _params{
      {"Wire", SHCCSTR("The wire to optimize and evolve."), {CoreInfo::WireType}},
      {"Fitness",
       SHCCSTR("The fitness wire to run at the end of the main wire evaluation "
               "and using its last output; should output a Float fitness value."),
       {CoreInfo::WireType}},
      {"Population", SHCCSTR("The population size."), {CoreInfo::IntType}},
      {"Mutation", SHCCSTR("The rate of mutation, 0.1 = 10%."), {CoreInfo::FloatType}},
      {"Crossover", SHCCSTR("The rate of crossover, 0.1 = 10%."), {CoreInfo::FloatType}},
      {"Extinction", SHCCSTR("The rate of extinction, 0.1 = 10%."), {CoreInfo::FloatType}},
      {"Elitism", SHCCSTR("The rate of elitism, 0.1 = 10%."), {CoreInfo::FloatType}},
      {"Threads", SHCCSTR("The number of cpu threads to use."), {CoreInfo::IntType}},
      {"Coroutines", SHCCSTR("The number of coroutines to run on each thread."), {CoreInfo::IntType}}};
  static inline Types _outputTypes{{CoreInfo::FloatType, CoreInfo::WireType}};
  static inline Type _outputType{{SHType::Seq, {.seqTypes = _outputTypes}}};

  std::unique_ptr<tf::Executor> _exec;

  OwnedVar _baseWire{};
  OwnedVar _fitnessWire{};
  std::vector<SHVar> _result;
  std::vector<Individual> _population;
  std::vector<Individual *> _sortedPopulation;
  std::vector<std::tuple<Individual *, Individual *, Individual *>> _crossingOver;
  int64_t _popsize = 64;
  int64_t _coros = 8;
  int64_t _threads = 2;
  double _mutation = 0.2;
  double _crossover = 0.2;
  double _extinction = 0.1;
  size_t _nkills = 0;
  double _elitism = 0.1;
  size_t _nelites = 0;
  size_t _era = 0;
};

struct Mutant {
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  static SHOptionalString help() {
    return SHCCSTR("This shard modifies the parameters of the shard specified in the Shard parameter. Which parameters are "
                   "modified is specified Indices parameter and how they are modified is specified in the Mutations parameter. "
                   "This shard should be used in conjunction with the Evolve shard to evolve the wire.");
  }
  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _shard = value;
      break;
    case 1:
      _indices = value;
      break;
    case 2: {
      _mutations = value;
      if (_mutations.valueType == SHType::Seq) {
        for (auto &mut : _mutations) {
          if (mut.valueType == SHType::ShardRef) {
            auto blk = mut.payload.shardValue;
            blk->owned = true;
          } else if (mut.valueType == SHType::Seq) {
            for (auto &bv : mut) {
              auto blk = bv.payload.shardValue;
              blk->owned = true;
            }
          }
        }
      }
    } break;
    case 3:
      _options = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _shard;
    case 1:
      return _indices;
    case 2:
      return _mutations;
    case 3:
      return _options;
    default:
      return SHVar();
    }
  }

  void cleanupMutations(SHContext *context) const {
    if (_mutations.valueType == SHType::Seq) {
      for (auto &mut : _mutations) {
        if (mut.valueType == SHType::ShardRef) {
          auto blk = mut.payload.shardValue;
          blk->cleanup(blk, context);
        } else if (mut.valueType == SHType::Seq) {
          for (auto &bv : mut) {
            auto blk = bv.payload.shardValue;
            blk->cleanup(blk, context);
          }
        }
      }
    }
  }

  void warmupMutations(SHContext *ctx) const {
    if (_mutations.valueType == SHType::Seq) {
      for (auto &mut : _mutations) {
        if (mut.valueType == SHType::ShardRef) {
          auto blk = mut.payload.shardValue;
          if (blk->warmup)
            blk->warmup(blk, ctx);
        } else if (mut.valueType == SHType::Seq) {
          for (auto &bv : mut) {
            auto blk = bv.payload.shardValue;
            if (blk->warmup)
              blk->warmup(blk, ctx);
          }
        }
      }
    }
  }

  void cleanup(SHContext *context) {
    _shard.cleanup(context);
    cleanupMutations(context);
  }

  void destroy() {}

  void warmup(SHContext *ctx) { _shard.warmup(ctx); }

  SHTypeInfo compose(SHInstanceData &data) {
    auto inner = mutant();
    // validate parameters
    if (_mutations.valueType == SHType::Seq && inner) {
      auto dataCopy = data;
      int idx = 0;
      auto innerParams = inner->parameters(inner);
      for (auto &mut : _mutations) {
        if (idx >= int(innerParams.len))
          break;
        TypeInfo ptype(inner->getParam(inner, idx), data);
        dataCopy.inputType = ptype;
        if (mut.valueType == SHType::ShardRef) {
          auto blk = mut.payload.shardValue;
          if (blk->compose) {
            auto res0 = blk->compose(blk, &dataCopy);
            if (res0.error.code != SH_ERROR_NONE) {
              std::string_view err(res0.error.message.string, size_t(res0.error.message.len));
              throw ComposeError(err);
            }
            auto res = res0.result;
            if (res != ptype) {
              throw SHException("Expected same type as input in parameter "
                                "mutation wire's output.");
            }
          }
        } else if (mut.valueType == SHType::Seq) {
          auto res = composeWire(mut.payload.seqValue, dataCopy);
          if (res.outputType != ptype) {
            throw SHException("Expected same type as input in parameter "
                              "mutation wire's output.");
          }
          arrayFree(res.exposedInfo);
          arrayFree(res.requiredInfo);
        }
        idx++;
      }
    }

    return _shard.compose(data).outputType;
  }

  SHExposedTypesInfo exposedVariables() {
    if (!_shard)
      return {};

    auto blks = _shard.shards();
    return blks.elements[0]->exposedVariables(blks.elements[0]);
  }

  SHExposedTypesInfo requiredVariables() {
    if (!_shard)
      return {};

    auto blks = _shard.shards();
    return blks.elements[0]->exposedVariables(blks.elements[0]);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHVar output{};
    _shard.activate(context, input, output);
    return output;
  }

  Shard *mutant() const {
    if (!_shard)
      return nullptr;

    auto blks = _shard.shards();
    return blks.elements[0];
  }

private:
  friend struct Evolve;
  ShardsVar _shard{};
  OwnedVar _indices{};
  OwnedVar _mutations{};
  OwnedVar _options{};
  static inline Parameters _params{
      {"Shard", SHCCSTR("The shard to mutate."), {CoreInfo::ShardRefType}},
      {"Indices", SHCCSTR("The parameter indices to mutate of the inner shard."), {CoreInfo::IntSeqType}},
      {"Mutations",
       SHCCSTR("Optional operations to apply to the parameters, if empty a default operator will be used."),
       {CoreInfo::ShardsOrNoneSeq}},
      {"Options",
       SHCCSTR("Mutation options table - a table with mutation options."),
       {CoreInfo::NoneType, CoreInfo::AnyTableType}}};
};

inline void mutateVar(SHVar &var) {
  switch (var.valueType) {
  case SHType::Int: {
    const auto add = Random::nextNormal1() * double(var.payload.intValue);
    var.payload.intValue += int64_t(add);
  } break;
  case SHType::Int2: {
    for (auto i = 0; i < 2; i++) {
      const auto add = Random::nextNormal1() * double(var.payload.int2Value[i]);
      var.payload.int2Value[i] += int64_t(add);
    }
  } break;
  case SHType::Int3: {
    for (auto i = 0; i < 3; i++) {
      const auto add = Random::nextNormal1() * double(var.payload.int3Value[i]);
      var.payload.int3Value[i] += int32_t(add);
    }
  } break;
  case SHType::Int4: {
    for (auto i = 0; i < 4; i++) {
      const auto add = Random::nextNormal1() * double(var.payload.int4Value[i]);
      var.payload.int4Value[i] += int32_t(add);
    }
  } break;
  case SHType::Int8: {
    for (auto i = 0; i < 8; i++) {
      const auto add = Random::nextNormal1() * double(var.payload.int8Value[i]);
      var.payload.int8Value[i] += int16_t(add);
    }
  } break;
  case SHType::Int16: {
    for (auto i = 0; i < 16; i++) {
      const auto add = Random::nextNormal1() * double(var.payload.int16Value[i]);
      var.payload.int16Value[i] += int8_t(add);
    }
  } break;
  case SHType::Float: {
    var.payload.floatValue += Random::nextNormal();
  } break;
  case SHType::Float2: {
    for (auto i = 0; i < 2; i++)
      var.payload.float2Value[i] += Random::nextNormal();
  } break;
  case SHType::Float3: {
    for (auto i = 0; i < 3; i++)
      var.payload.float3Value[i] += float(Random::nextNormal());
  } break;
  case SHType::Float4: {
    for (auto i = 0; i < 4; i++)
      var.payload.float4Value[i] += float(Random::nextNormal());
  } break;
  default:
    break;
  }
}

inline void Evolve::gatherMutants(SHWire *wire, std::vector<MutantInfo> &out) {
  std::vector<ShardInfo> shards;
  gatherShards(wire, shards);
  auto pos = std::remove_if(std::begin(shards), std::end(shards), [](ShardInfo &info) { return info.name != "Mutant"; });
  if (pos != std::end(shards)) {
    std::for_each(std::begin(shards), pos, [&](ShardInfo &info) {
      auto mutator = reinterpret_cast<const ShardWrapper<Mutant> *>(info.shard);
      auto &minfo = out.emplace_back(mutator->shard);
      auto mutant = mutator->shard.mutant();
      if (mutator->shard._indices.valueType == SHType::Seq) {
        for (auto &idx : mutator->shard._indices) {
          auto i = int(idx.payload.intValue);
          minfo.originalParams.emplace_back(i, mutant->getParam(mutant, i));
        }
      }
    });
  }
}

inline void Evolve::crossover(Individual &child, const Individual &parent0, const Individual &parent1) {
  auto cmuts = child.mutants.cbegin();
  auto p0muts = parent0.mutants.cbegin();
  auto p1muts = parent1.mutants.cbegin();
  for (; cmuts != child.mutants.end() && p0muts != parent0.mutants.end() && p1muts != parent1.mutants.end();
       ++cmuts, ++p0muts, ++p1muts) {
    auto cb = cmuts->shard.get().mutant();
    auto p0b = p0muts->shard.get().mutant();
    auto p1b = p1muts->shard.get().mutant();
    if (cb && p0b && p1b) {
      // use crossover proc if avail
      if (cb->crossover) {
        auto p0s = p0b->getState(p0b);
        auto p1s = p1b->getState(p1b);
        cb->crossover(cb, &p0s, &p1s);
      }
      // check if we have mutant params and cross them over
      auto &indices = cmuts->shard.get()._indices;
      if (indices.valueType == SHType::Seq) {
        for (auto &idx : indices) {
          const auto i = int(idx.payload.intValue);
          const auto r = Random::nextDouble();
          if (r < 0.33) {
            // take from 0
            auto val = p0b->getParam(p0b, i);
            cb->setParam(cb, i, &val);
          } else if (r < 0.66) {
            // take from 1
            auto val = p1b->getParam(p1b, i);
            cb->setParam(cb, i, &val);
          } // else we keep
        }
      }
    }
  }
}

inline void Evolve::mutate(Evolve::Individual &individual) {
  auto wire = SHWire::sharedFromRef(individual.wire.payload.wireValue);
  // we need to hack this in as we run out of context
  Coroutine foo{};
  SHFlow flow{0, nullptr, false};
  SHContext ctx(&foo, wire.get(), &flow);
  ctx.wireStack.push_back(wire.get());
  std::for_each(std::begin(individual.mutants), std::end(individual.mutants), [&](MutantInfo &info) {
    if (Random::nextDouble() > _mutation) {
      return;
    }

    auto &mutator = info.shard.get();
    auto &options = mutator._options;
    if (info.shard.get().mutant()) {
      auto mutant = info.shard.get().mutant();
      auto &indices = mutator._indices;
      if (mutant->mutate && (indices.valueType == SHType::None || rand() < 0.5)) {
        // In the case the shard has `mutate`
        auto table = options.valueType == SHType::Table ? options.payload.tableValue : SHTable();
        mutant->mutate(mutant, table);
      } else if (indices.valueType == SHType::Seq) {
        auto &iseq = indices.payload.seqValue;
        // do stuff on the param
        // select a random one
        auto rparam = Random::nextInt() % iseq.len;
        auto current = mutant->getParam(mutant, int(iseq.elements[rparam].payload.intValue));
        // if we have mutation shards use them
        // if not use default operation
        if (mutator._mutations.valueType == SHType::Seq && uint32_t(rparam) < mutator._mutations.payload.seqValue.len) {
          // we need to warmup / cleanup in this case
          // mutant mini wire also currently is not composed! FIXME?
          mutator.warmupMutations(&ctx);
          auto mblks = mutator._mutations.payload.seqValue.elements[rparam];
          if (mblks.valueType == SHType::ShardRef) {
            auto blk = mblks.payload.shardValue;
            current = *blk->activate(blk, &ctx, &current);
          } else if (mblks.valueType == SHType::Seq) {
            auto blks = mblks.payload.seqValue;
            SHVar out{};
            activateShards(blks, &ctx, current, out);
            current = out;
          } else {
            // Was likely None, so use default op
            mutateVar(current);
          }
          mutator.cleanupMutations(nullptr);
        } else {
          mutateVar(current);
        }
        mutant->setParam(mutant, int(iseq.elements[rparam].payload.intValue), &current);
      }
    }
  });
}

inline void Evolve::resetState(Evolve::Individual &individual) {
  // Reset params and state
  std::for_each(std::begin(individual.mutants), std::end(individual.mutants), [](MutantInfo &info) {
    auto mutant = info.shard.get().mutant();
    if (!mutant)
      return;

    if (mutant->resetState)
      mutant->resetState(mutant);

    for (auto [idx, val] : info.originalParams) {
      mutant->setParam(mutant, idx, &val);
    }
  });
}

struct DShard {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard is a dynamic shard that will transform into the shard specified in the Name parameter at runtime. This shard "
        "can be used in conjunction with Mutate and Evolve to create complex genetic algorithm implementations.");
  }

  static inline Parameters _params{{"Name", SHCCSTR("The name of the shard to transform into."), {CoreInfo::StringType}},
                                   {"Parameters", SHCCSTR("The parameters to pass to the new shard."), {CoreInfo::AnySeqType}}};

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0: {
      _name = SHSTRVIEW(value);

      // destroy if we had a shard already
      if (_wrapped)
        _wrapped->destroy(_wrapped);
      // create the shard directly here
      _wrapped = createShard(_name.c_str());
      // and setup if successful
      if (_wrapped) {
        incRef(_wrapped);
        _wrapped->setup(_wrapped);
      }
    } break;
    case 1: {
      _wrappedParams.clear();
      for (auto &v : value) {
        _wrappedParams.emplace_back(v);
      }
    } break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return shards::Var(_name);
    case 1: {
      SHVar res{};
      res.valueType = SHType::Seq;
      const auto nparams = _wrappedParams.size();
      res.payload.seqValue.elements = nparams > 0 ? &_wrappedParams.front() : nullptr;
      res.payload.seqValue.len = nparams;
      res.payload.seqValue.cap = 0;
      return res;
    }
    default:
      return {};
    }
  }

  SHTypeInfo compose(SHInstanceData &data) {
    if (!_wrapped)
      return {};

    // set the wrapped params here
    auto params = _wrapped->parameters(_wrapped);
    for (uint32_t i = 0; i < params.len; i++) {
      if (!validateSetParam(_wrapped, i, _wrappedParams[i]))
        throw SHException("Failed to validate a parameter within a wrapped DShard.");
      _wrapped->setParam(_wrapped, int(i), &_wrappedParams[i]);
    }
    // and compose finally
    if (_wrapped->compose) {
      auto res = _wrapped->compose(_wrapped, &data);
      if (res.error.code != 0) {
        throw SHException("Failed to compose a wrapped DShard.");
      }
      return res.result;
    } else {
      // need to return something valid following runtime validation
      auto outputTypes = _wrapped->outputTypes(_wrapped);
      if (outputTypes.len == 1 && outputTypes.elements[0].basicType != SHType::Any) {
        return outputTypes.elements[0];
      } else {
        return {};
      }
    }
  }

  void destroy() {
    if (_wrapped) {
      decRef(_wrapped);
      _wrapped = nullptr;
    }
  }

  SHTypesInfo inputTypes() {
    if (_wrapped)
      return _wrapped->inputTypes(_wrapped);
    else
      return CoreInfo::NoneType;
  }

  SHTypesInfo outputTypes() {
    if (_wrapped)
      return _wrapped->outputTypes(_wrapped);
    else
      return CoreInfo::NoneType;
  }

  SHExposedTypesInfo exposedVariables() {
    if (_wrapped)
      return _wrapped->exposedVariables(_wrapped);
    else
      return {};
  }

  SHExposedTypesInfo requiredVariables() {
    if (_wrapped)
      return _wrapped->requiredVariables(_wrapped);
    else
      return {};
  }

  void warmup(SHContext *context) {
    if (_wrapped && _wrapped->warmup) // it's optional!
      _wrapped->warmup(_wrapped, context);
  }

  void cleanup(SHContext *context) {
    if (_wrapped)
      _wrapped->cleanup(_wrapped, context);
  }

  const SHVar &activate(SHContext *context, const SHVar &input) {
    if (!_wrapped) {
      throw ActivationError("Wrapped shard was null!");
    }

    return *_wrapped->activate(_wrapped, context, &input);
  }

  void mutate(SHTable options) {
    if (_wrapped && _wrapped->mutate)
      _wrapped->mutate(_wrapped, options);
  }

  void crossover(SHVar state0, SHVar state1) {
    if (_wrapped && _wrapped->crossover)
      _wrapped->crossover(_wrapped, &state0, &state1);
  }

  SHVar getState() {
    if (_wrapped && _wrapped->getState)
      return _wrapped->getState(_wrapped);
    else
      return {};
  }

  void setState(SHVar state) {
    if (_wrapped && _wrapped->setState)
      _wrapped->setState(_wrapped, &state);
  }

  void resetState() {
    if (_wrapped && _wrapped->resetState)
      _wrapped->resetState(_wrapped);
  }

private:
  Shard *_wrapped = nullptr;
  std::string _name;
  std::vector<OwnedVar> _wrappedParams;
};

} // namespace Genetic
} // namespace shards

SHARDS_REGISTER_FN(genetic) {
  using namespace shards::Genetic;
  REGISTER_SHARD("Evolve", Evolve);
  REGISTER_SHARD("Mutant", Mutant);
  REGISTER_SHARD("DShard", DShard);
}
