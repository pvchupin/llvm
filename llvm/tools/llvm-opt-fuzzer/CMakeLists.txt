set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  BitReader
  BitWriter
  CodeGen
  Core
  Coroutines
  IPO
  IRReader
  AggressiveInstCombine
  InstCombine
  Instrumentation
  FuzzerCLI
  FuzzMutate
  MC
  ObjCARCOpts
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  Passes
  SYCLLowerIR
)

add_llvm_fuzzer(llvm-opt-fuzzer
  llvm-opt-fuzzer.cpp
  DUMMY_MAIN DummyOptFuzzer.cpp
  )
