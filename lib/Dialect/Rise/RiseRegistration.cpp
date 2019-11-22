//
// Created by martin on 22/11/2019.
//

#include "mlir/Dialect/Rise/Dialect.h"
using namespace mlir;

// Static initialization for loop dialect registration.
static DialectRegistration<rise::RiseDialect> Rise;