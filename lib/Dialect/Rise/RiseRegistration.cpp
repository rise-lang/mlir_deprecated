//
// Created by martin on 22/11/2019.
//

#include "mlir/Dialect/Rise/Dialect.h"
using namespace mlir;
using namespace mlir::rise;

// Static initialization for RISE dialect registration.
static DialectRegistration<rise::RiseDialect> Rise;