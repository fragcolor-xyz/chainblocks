#ifndef C2569599_BC02_4BCE_BE27_6A08E1D736DA
#define C2569599_BC02_4BCE_BE27_6A08E1D736DA

#include <gfx/data_cache/data_cache.hpp>

namespace gfx::data {
std::shared_ptr<data::IDataCacheIO> createShardsDataCacheIO(std::string_view rootPath);
std::shared_ptr<data::IDataCacheIO> getDefaultDataCacheIO();
} // namespace gfx

#endif /* C2569599_BC02_4BCE_BE27_6A08E1D736DA */
