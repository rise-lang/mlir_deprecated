//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToStandard/ConvertRiseToStandard.h"
#include "mlir/Dialect/Rise/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include <iostream>

using namespace mlir;
using namespace mlir::rise;


namespace {
struct ConvertRiseToStandardPass : public ModulePass<ConvertRiseToStandardPass> {
    void runOnModule() override;
};
} // namespace


Type RiseTypeConverter::convertType(Type t) {
    if (t.getKind() == RiseTypeKind::RISE_INT) {
        return convertIntType(t.cast<Int>());
    }
    return t;
}

Type RiseTypeConverter::convertIntType(Int type) {
//    StandardOpsDialect::
}








//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct LiteralLowering : public OpRewritePattern<LiteralOp> {
    using OpRewritePattern<LiteralOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(LiteralOp literalOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult
LiteralLowering::matchAndRewrite(LiteralOp literalOp, PatternRewriter &rewriter) const {
    Location loc = literalOp.getLoc();
    LiteralAttr literalAttr = literalOp.getAttrOfType<LiteralAttr>("literalValue");

    ///conversion to itself
//    rewriter.create<LiteralOp>(loc, literalAttr.getType(), literalAttr);
//    rewriter.eraseOp(literalOp);

    int32_t value = std::stoi(literalAttr.getValue());
    rewriter.create<ConstantIntOp>(loc, value, 32); //why is this getting such a strange SSA id?
    rewriter.eraseOp(literalOp);
    return matchSuccess();
}







///gather all patterns
void mlir::populateRiseToStdConversionPatterns(
        OwningRewritePatternList &patterns, MLIRContext *ctx) {
    patterns.insert<LiteralLowering>(ctx);
}


//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ConvertRiseToStandardPass::runOnModule() {
    auto module = getModule();

    // Convert to the LLVM IR dialect using the converter defined above.
    OwningRewritePatternList patterns;
//    LinalgTypeConverter converter(&getContext());
//    populateAffineToStdConversionPatterns(patterns, &getContext());
//    populateLoopToStdConversionPatterns(patterns, &getContext());
//    populateStdToLLVMConversionPatterns(converter, patterns);
//    populateVectorToLLVMConversionPatterns(converter, patterns);
//    populateLinalgToStandardConversionPatterns(patterns, &getContext());
//    populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());
    populateRiseToStdConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
//    target.addDynamicallyLegalOp<FuncOp>(
//            [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
//    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    if (failed(applyPartialConversion(module, target, patterns)))
        signalPassFailure();
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::rise::createConvertRiseToStandardPass() {
    return std::make_unique<ConvertRiseToStandardPass>();
}

static PassRegistration<ConvertRiseToStandardPass>
        pass("convert-rise-to-standard",
             "Convert the operations from the rise dialect into the standard dialect");
