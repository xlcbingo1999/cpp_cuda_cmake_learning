
#pragma once

#include "core/debugbreak.h"
#include "core/log.h"
#include "core/log_tag.h"
#include <cassert>
#include <string>

/**
 * @addtogroup utilities
 * @{
 */

/**
 * @brief like c `assert()`. log fatal error message, and auto stop program if
 * you are debugging
 * @param expr expression, be false will assert
 * @param msg  log fatal messge while assert
 */
#define Assert(expr, msg)        \
    do {                         \
        if (!(expr)) {           \
            LOGF("Assert", msg); \
            debug_break();       \
        }                        \
    } while (0)

/**
 * @brief tag a code not finish, run this macro will break app and step into
 * debug mode
 */
#define TODO(msg) Assert(false, msg)

/**
 * @}
 */