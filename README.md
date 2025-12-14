# llama-cpp-experimental

This is an experimental fork of the original llama.cpp.

The purpose of this repo is to give real world examples how you can modify different aspects of the llama.cpp code base to enable things like:
- Custom quantization schemes
- Custom model infernce architectures
- Custom data types
- Custom GGUF generators
- Custom device backends with CPU fallback
- and more...

Every branch in this has one example implementation. 

The user can diff this branch against the main branch to understand what changes one needs to perform to get what you want. 

The documentation of llama.cpp is very sparse without any real world examples how to modify the code base directly. Multiple changes in different places, etc. The tools and methods also change over time, so i will try to keep this examples up to date with mainstream. 

I hope it helps you explore codebase of llama.cpp a bit easier. 

Please dont use any code for production as its purely educational, not optimized and no safety taken into account. 

This branch `testing` has all examples combined into each other and is the most up to date branch. 

Every other branch will have suffix `example_{tool/method/tutorial}`. 

**Good luck ;D**