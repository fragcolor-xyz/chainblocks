# Shards: Programming That Flows

<img src="assets/wide-logo.png">

<div align="center">

  <a href="">[![license](https://img.shields.io/github/license/fragcolor-xyz/shards)](./LICENSE)</a>
  <a href="">[![CI](https://github.com/fragcolor-xyz/shards/actions/workflows/main.yml/badge.svg)](https://github.com/fragcolor-xyz/shards/actions/workflows/main.yml)</a>
  <a href="">[![codecov](https://codecov.io/gh/fragcolor-xyz/shards/branch/devel/graph/badge.svg?token=4PMT2FQFDS)](https://codecov.io/gh/fragcolor-xyz/shards)</a>
  <a href="">[![docs](https://img.shields.io/badge/docs-8A2BE2)](https://docs.fragcolor.com/)</a>

</div>

Shards is a revolutionary programming language that mirrors how data naturally moves in real systems. Unlike traditional languages that treat data like static containers, Shards embraces the dynamic flow of information - making it intuitive, powerful, and uniquely suited for both visual and textual programming.

## Why Shards?

Imagine programming as directing streams of data, like a master netrunner orchestrating information flows. That's Shards:

```shards
; Simple, intuitive data flow
["Hello" name "!"] | String.Join | Log

; Powerful concurrent processing
@wire(background {
    input | Transform | Process > result
    result | Network.Send
})
```

### Core Principles

1. **Everything Flows** - Data moves naturally through transformations:
   - No artificial variable assignments
   - Intuitive pipe operator (`|`) shows data direction
   - Clean, readable code that mirrors real data processing

2. **Visual & Textual Harmony**
   - Perfect 1:1 mapping between visual blocks and code
   - No "visual programming spaghetti"
   - Natural flow makes both forms equally intuitive

3. **Smart & Safe**
   - Automatic type inference
   - Compile-time safety
   - High-performance execution

4. **True Concurrency**
   - Built-in wire system for parallel processing
   - Lightweight, efficient scheduling
   - Natural handling of complex async operations

## Quick Start

```shards
; Create a mutable flow channel
5 >= counter

; Transform the flow
counter | Math.Add(1) > counter

; Chain multiple transformations
input | Transform | Process | Network.Send
```

## Real-World Example

Here's a simple UI application that shows Shards in action:

```shards
@wire(main-loop {
    GFX.MainWindow(
        Title: "Flow Demo"
        Contents: {
            UI.Button("Click Me!" {
                "Button clicked!" | Log
                counter | Math.Add(1) > counter
            })
            
            counter | ToString | UI.Label
        }
    )
} Looped: true)

@mesh(root)
@schedule(root main-loop)
@run(root FPS: 60)
```

## Key Features

- **Multi-Platform**: Runs everywhere - desktop, mobile, web
- **High-Performance**: Optimized C++ and Rust core
- **Visual Development**: First-class support for visual programming
- **Rich Standard Library**: 828+ built-in shards for common tasks
- **Game Development**: Perfect for real-time interactive applications
- **AI-Ready**: Powers next-gen AI-assisted development tools

## Perfect For

- Game Development
- Interactive Applications
- Data Processing
- System Integration
- Visual Programming
- Real-time Systems

## Get Started

1. [Install Shards](https://docs.fragcolor.com/contribute/getting-started/)
2. [Read the Documentation](https://docs.fragcolor.com)
3. [Join our Community](https://discord.gg/fragcolor)

## Technical Details

- **Type System**: Strong, inferred, compile-time checked
- **Concurrency**: Built-in wire system with cooperative scheduling
- **Performance**: Zero-overhead abstractions
- **Memory**: Safe, efficient memory management
- **Interop**: Easy integration with C++, Rust, and other languages

## License

BSD 3-Clause license - Free for personal and commercial use.
