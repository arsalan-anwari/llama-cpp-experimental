# llama-cpp-experimental

This is an experimental fork of the original llama.cpp.

The purpose of this repo is to give real-world examples of how you can modify different aspects of the llama.cpp code base to enable things like:
- Custom quantization schemes
- Custom model inference architectures
- Custom data types
- Custom GGUF generators
- Custom device backends with CPU fallback
- and more...

Every branch in this has one example implementation. 

The user can diff this branch against the main branch to understand what changes one needs to perform to get what you want. 

The documentation of llama.cpp is very sparse without any real-world examples of how to modify the code base directly. Multiple changes in different places, etc. 
The tools and methods also change over time, so I will try to keep these examples up to date with mainstream. 

I hope it helps you explore the codebase of `llama.cpp` a bit easier. 

Please don't use any code for production, as it's purely educational, not optimized, and no safety is taken into account. 

The branch `testing` has all examples combined into each other and is the most up to date branch. 
The main branch is a snapshot of the upstream of `llama.cpp`. (last update: 14-12-2025 01:15 CEST)  

Every other branch will have a suffix `example_{tool/method/tutorial}`. 

**Good luck ;D**
