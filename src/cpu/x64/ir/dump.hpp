/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_X64_IR_DUMP_HPP
#define CPU_X64_IR_DUMP_HPP

// Debug output of the IR pipeline.
//
// A kernel built from the IR calls `print_kernel_dump()` at the end of its
// `generate()`. The call prints a text description of the kernel. It prints
// only in dev-mode builds, and only when `ONEDNN_VERBOSE` holds the `x64ir`
// token. The output starts with a summary: the kernel name, the ISA, the code
// size, one line per register file, and the spilled virtual registers with the
// operations that access their stack slots. An empty line and the IR dump
// follow, one line per operation, with the location of each virtual register
// and the register pressure at each operation.
//
// The output uses only generic pipeline objects, such as the generator, the
// IR, and the static data. It does not depend on a particular builder.
//
// The output does not depend on `debuginfo=` or on `all`. `all` enables the
// standard verbose output, and this backend-specific dump should not mix with
// it.
//
// The output for one kernel is formatted into one string and printed with a
// single stdio call, so the output of kernels created at the same time does
// not mix. It starts and ends with a marker line:
//
//   begin x64ir <kernel name> isa=<isa>
//   ...
//   end x64ir
//
// The formatting functions are compiled in every build, so the unit tests run
// without dev mode. Only the check in `print_kernel_dump()` depends on dev
// mode.

#include <string>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Returns true when `verbose_value`, a comma-separated `ONEDNN_VERBOSE` value,
// holds the token `x64ir`.
//
// Export for testing.
bool DNNL_API has_x64ir_token(const std::string &verbose_value);

// Kernel facts that the IR does not hold. `print_kernel_dump()` fills them in.
//
//   name      - kernel name, as returned by `name()`
//   isa       - ISA the kernel is generated for (`max_cpu_isa()`)
//   code_size - total code size in bytes at the end of `generate()`,
//               including static data
//   data_size - static data in bytes, written after the postamble (see
//               `data_section_t`)
//   reg_cfg   - register configuration that the allocator used
//   alloc     - register allocation of the IR
//
// `reg_cfg` and `alloc` must not be null.
struct kernel_info_t {
    const char *name = "";
    cpu_isa_t isa = isa_undef;
    size_t code_size = 0;
    size_t data_size = 0;
    const reg_config_t *reg_cfg = nullptr;
    const reg_alloc_result_t *alloc = nullptr;
};

// Returns the IR dump of `ir`, one line per operation. Each line starts with
// the operation index and is indented by loop depth.
//
// Operands print as follows:
//   g<id>, m<id>     - gpr and mask vreg with id `<id>`
//   <dt>:v<id>       - vec vreg with id `<id>` that holds data type `<dt>`
//   [g<id>+<disp>]   - memory operand with a decimal byte offset.
//                      `[param+<disp>]` reads the kernel argument struct. A
//                      vector access is prefixed with the data type in memory
//                      (`bf16:[g3+0]`).
//   L<id>            - IR label with id `<id>`
//
// Export for testing.
std::string DNNL_API to_string(const ir_t &ir);

// Returns the IR dump of `ir` with the register allocation in `info`. The dump
// starts with a header line. Each operation line has one more column, between
// the index and the operation, with the register pressure of each register
// file at the operation. Each vreg operand is followed by its location:
//   g<id>@<reg>             - the vreg is in the physical register `<reg>`, as
//                             the emitter names it (`rax`, `ymm3`, `zmm3`, or
//                             `k1`)
//   g<id>@[rsp+<off>](temp <reg>)
//                           - the vreg is spilled to the stack slot at the
//                             decimal byte offset `<off>`, and the operation
//                             moves it through the temp register `<reg>`.
//                             `(no temp)` means that the operation has no temp
//                             for it, so the emitter cannot lower the
//                             operation.
//
// Export for testing.
std::string DNNL_API to_string(const ir_t &ir, const kernel_info_t &info);

// Returns the output for one kernel, framed with the marker lines.
//
// Export for testing.
std::string DNNL_API format_kernel_dump(
        const kernel_info_t &info, const ir_t &ir);

// Prints the output for the kernel that `gen` generated from `ir`, with the
// static data in `data` and the register allocation `alloc` from `reg_cfg`.
// Prints nothing in builds without dev mode, or when `ONEDNN_VERBOSE` does not
// hold the `x64ir` token. The variable is read once per process.
//
// The kernel calls it at the end of `generate()`, after the static data is
// written, so the code size includes the static data.
void print_kernel_dump(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data, const reg_config_t &reg_cfg,
        const reg_alloc_result_t &alloc);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
