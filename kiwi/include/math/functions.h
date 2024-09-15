#ifndef KIWI_MATH_FUNCTIONS_H
#define KIWI_MATH_FUNCTIONS_H

#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include "parsing/tokens.h"
#include "tracing/error.h"
#include "typing/value.h"
#include "rng.h"

static k_string get_string(
    const Token& term, const k_value& arg,
    const k_string& message = "Expected a string value.") {
  if (!std::holds_alternative<k_string>(arg)) {
    throw ConversionError(term, message);
  }
  return std::get<k_string>(arg);
}

static k_int get_integer(
    const Token& term, const k_value& arg,
    const k_string& message = "Expected an integer value.") {
  if (std::holds_alternative<double>(arg)) {
    return static_cast<k_int>(std::get<double>(arg));
  }
  if (!std::holds_alternative<k_int>(arg)) {
    throw ConversionError(term, message);
  }
  return std::get<k_int>(arg);
}

static double get_double(
    const Token& term, const k_value& arg,
    const k_string& message = "Expected an integer or double value.") {
  if (std::holds_alternative<k_int>(arg)) {
    return static_cast<double>(std::get<k_int>(arg));
  } else if (std::holds_alternative<double>(arg)) {
    return std::get<double>(arg);
  }

  throw ConversionError(term, message);
}

struct {
  bool is_zero(const Token& term, const k_value& v) {
    if (std::holds_alternative<double>(v)) {
      return std::get<double>(v) == 0.0;
    } else if (std::holds_alternative<k_int>(v)) {
      return std::get<k_int>(v) == 0;
    }

    throw ConversionError(term,
                          "Cannot check non-numeric value for zero value.");
  }

  k_value do_addition(const Token& token, const k_value& left,
                      const k_value& right) {
    k_value result;

    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      result = std::get<k_int>(left) + std::get<k_int>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<double>(right)) {
      result = std::get<double>(left) + std::get<double>(right);
    } else if (std::holds_alternative<k_int>(left) &&
               std::holds_alternative<double>(right)) {
      result =
          static_cast<double>(std::get<k_int>(left)) + std::get<double>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<k_int>(right)) {
      result =
          std::get<double>(left) + static_cast<double>(std::get<k_int>(right));
    } else if (std::holds_alternative<k_string>(right)) {
      std::ostringstream build;
      if (std::holds_alternative<k_int>(left)) {
        build << std::get<k_int>(left);
      } else if (std::holds_alternative<double>(left)) {
        build << std::get<double>(left);
      } else if (std::holds_alternative<bool>(left)) {
        build << std::boolalpha << std::get<bool>(left);
      } else if (std::holds_alternative<k_string>(left)) {
        build << std::get<k_string>(left);
      }

      build << std::get<k_string>(right);

      result = build.str();
    } else if (std::holds_alternative<k_string>(left)) {
      std::ostringstream build;
      build << std::get<k_string>(left);

      if (std::holds_alternative<k_int>(right)) {
        build << std::get<k_int>(right);
      } else if (std::holds_alternative<double>(right)) {
        build << std::get<double>(right);
      } else if (std::holds_alternative<bool>(right)) {
        build << std::boolalpha << std::get<bool>(right);
      } else if (std::holds_alternative<k_string>(right)) {
        build << std::get<k_string>(right);
      }

      result = build.str();
    } else if (std::holds_alternative<k_list>(left)) {
      auto list = std::get<k_list>(left);
      if (std::holds_alternative<k_list>(right)) {
        const auto& rightList = std::get<k_list>(right)->elements;
        for (const auto& item : rightList) {
          list->elements.emplace_back(item);
        }
      } else {
        list->elements.emplace_back(right);
      }
      return list;
    } else {
      throw ConversionError(token, "Conversion error in addition.");
    }

    return result;
  }

  k_value do_subtraction(const Token& token, const k_value& left,
                         const k_value& right) {
    k_value result;

    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      result = std::get<k_int>(left) - std::get<k_int>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<double>(right)) {
      result = std::get<double>(left) - std::get<double>(right);
    } else if (std::holds_alternative<k_int>(left) &&
               std::holds_alternative<double>(right)) {
      result =
          static_cast<double>(std::get<k_int>(left)) - std::get<double>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<k_int>(right)) {
      result =
          std::get<double>(left) - static_cast<double>(std::get<k_int>(right));
    } else if (std::holds_alternative<k_list>(left) &&
               !std::holds_alternative<k_list>(right)) {
      std::vector<k_value> listValues;
      const auto& leftList = std::get<k_list>(left)->elements;
      bool found = false;

      for (const auto& item : leftList) {
        if (!found && same_value(item, right)) {
          found = true;
          continue;
        }
        listValues.emplace_back(item);
      }

      return std::make_shared<List>(listValues);
    } else if (std::holds_alternative<k_list>(left) &&
               std::holds_alternative<k_list>(right)) {
      std::vector<k_value> listValues;
      const auto& leftList = std::get<k_list>(left)->elements;
      const auto& rightList = std::get<k_list>(right)->elements;

      for (const auto& item : leftList) {
        bool found = false;

        for (const auto& ritem : rightList) {
          if (same_value(item, ritem)) {
            found = true;
            break;
          }
        }

        if (!found) {
          listValues.emplace_back(item);
        }
      }
      return std::make_shared<List>(listValues);
    } else {
      throw ConversionError(token, "Conversion error in subtraction.");
    }

    return result;
  }

  k_value do_exponentiation(const Token& token, const k_value& left,
                            const k_value& right) {
    k_value result;

    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      result = static_cast<k_int>(
          pow(std::get<k_int>(left), std::get<k_int>(right)));
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<double>(right)) {
      result = pow(std::get<double>(left), std::get<double>(right));
    } else if (std::holds_alternative<k_int>(left) &&
               std::holds_alternative<double>(right)) {
      result = pow(static_cast<double>(std::get<k_int>(left)),
                   std::get<double>(right));
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<k_int>(right)) {
      result = pow(std::get<double>(left),
                   static_cast<double>(std::get<k_int>(right)));
    } else {
      throw ConversionError(token, "Conversion error in exponentiation.");
    }

    return result;
  }

  k_value do_modulus(const Token& token, const k_value& left,
                     const k_value& right) {
    k_value result;

    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      auto rhs = std::get<k_int>(right);
      if (rhs == 0) {
        throw DivideByZeroError(token);
      }
      result = std::get<k_int>(left) % std::get<k_int>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<double>(right)) {
      double rhs = std::get<double>(right);
      if (rhs == 0.0) {
        throw DivideByZeroError(token);
      }
      result = fmod(std::get<double>(left), rhs);
    } else if (std::holds_alternative<k_int>(left) &&
               std::holds_alternative<double>(right)) {
      double rhs = std::get<double>(right);
      if (rhs == 0.0) {
        throw DivideByZeroError(token);
      }
      result = fmod(std::get<k_int>(left), rhs);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<k_int>(right)) {
      double rhs = static_cast<double>(std::get<k_int>(right));
      if (rhs == 0) {
        throw DivideByZeroError(token);
      }
      result = fmod(std::get<double>(left), rhs);
    } else {
      throw ConversionError(token, "Conversion error in modulus.");
    }

    return result;
  }

  k_value do_division(const Token& token, const k_value& left,
                      const k_value& right) {
    k_value result;

    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      auto rhs = std::get<k_int>(right);
      if (rhs == 0) {
        throw DivideByZeroError(token);
      }
      result = std::get<k_int>(left) / rhs;
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<double>(right)) {
      double rhs = std::get<double>(right);
      if (rhs == 0.0) {
        throw DivideByZeroError(token);
      }
      result = std::get<double>(left) / rhs;
    } else if (std::holds_alternative<k_int>(left) &&
               std::holds_alternative<double>(right)) {
      double rhs = std::get<double>(right);
      if (rhs == 0.0) {
        throw DivideByZeroError(token);
      }
      result = static_cast<double>(std::get<k_int>(left)) / rhs;
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<k_int>(right)) {
      double rhs = static_cast<double>(std::get<k_int>(right));
      if (rhs == 0.0) {
        throw DivideByZeroError(token);
      }
      result = std::get<double>(left) / rhs;
    } else {
      throw ConversionError(token, "Conversion error in division.");
    }

    return result;
  }

  k_value do_multiplication(const Token& token, const k_value& left,
                            const k_value& right) {
    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      return std::get<k_int>(left) * std::get<k_int>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<double>(right)) {
      return std::get<double>(left) * std::get<double>(right);
    } else if (std::holds_alternative<k_int>(left) &&
               std::holds_alternative<double>(right)) {
      return static_cast<double>(std::get<k_int>(left)) *
             std::get<double>(right);
    } else if (std::holds_alternative<double>(left) &&
               std::holds_alternative<k_int>(right)) {
      return std::get<double>(left) *
             static_cast<double>(std::get<k_int>(right));
    } else if (std::holds_alternative<k_string>(left) &&
               std::holds_alternative<k_int>(right)) {
      return do_string_multiplication(left, right);
    } else if (std::holds_alternative<k_list>(left) &&
               std::holds_alternative<k_int>(right)) {
      return do_list_multiplication(token, left, right);
    }

    throw ConversionError(token, "Conversion error in multiplication.");
  }

  k_value do_list_multiplication(const Token& token, const k_value& left,
                                 const k_value& right) {
    auto list = std::get<k_list>(left);
    auto multiplier = std::get<k_int>(right);

    if (multiplier < 1) {
      throw SyntaxError(token,
                        "List multiplier must be a positive non-zero integer.");
    }

    if (list->elements.size() == 0) {
      throw SyntaxError(token, "Cannot multiply an empty list.");
    }

    auto newList = std::make_shared<List>();
    auto& elements = newList->elements;
    elements.reserve(list->elements.size() * multiplier);

    for (int i = 0; i < multiplier; ++i) {
      for (const auto& item : list->elements) {
        elements.emplace_back(clone_value(item));
      }
    }

    return newList;
  }

  k_value do_string_multiplication(const k_value& left, const k_value& right) {
    auto string = std::get<k_string>(left);
    auto multiplier = std::get<k_int>(right);

    if (multiplier <= 0) {
      return k_string();
    }

    k_string build;
    build.reserve(string.size() * multiplier);

    for (int i = 0; i < multiplier; ++i) {
      build.append(string);
    }

    return build;
  }

  bool is_truthy(const k_value& value) {
    switch (value.index()) {
      case 0:  // k_int
        return std::get<k_int>(value) != static_cast<k_int>(0);

      case 1:  // double
        return std::get<double>(value) != static_cast<double>(0);

      case 2:  // bool
        return std::get<bool>(value);

      case 3:  // k_string
        return !std::get<k_string>(value).empty();

      case 4:  // k_list
        return !std::get<k_list>(value)->elements.empty();

      case 5:  // k_hash
        return std::get<k_hash>(value)->size() > 0;

      case 6:  // k_object
        return true;

      case 7:  // k_lambda
        return true;

      case 8:  // k_null
        return false;

      default:
        return false;
    }
  }

  k_value do_eq_comparison(const k_value& left, const k_value& right) {
    return same_value(left, right);
  }

  k_value do_neq_comparison(const k_value& left, const k_value& right) {
    return !same_value(left, right);
  }

  k_value do_lt_comparison(const k_value& left, const k_value& right) {
    return lt_value(left, right);
  }

  k_value do_lte_comparison(const k_value& left, const k_value& right) {
    return lt_value(left, right) || same_value(left, right);
  }

  k_value do_gt_comparison(const k_value& left, const k_value& right) {
    return gt_value(left, right);
  }

  k_value do_gte_comparison(const k_value& left, const k_value& right) {
    return gt_value(left, right) || same_value(left, right);
  }

  k_value do_bitwise_and(const Token& token, const k_value& left,
                         const k_value& right) {
    if (std::holds_alternative<k_int>(left)) {
      auto lhs = std::get<k_int>(left);
      if (std::holds_alternative<k_int>(right)) {
        return lhs & std::get<k_int>(right);
      } else if (std::holds_alternative<double>(right)) {
        return lhs & static_cast<k_int>(std::get<double>(right));
      } else if (std::holds_alternative<bool>(right)) {
        k_int rhs = std::get<bool>(right) ? 1 : 0;
        return lhs & rhs;
      }
    }

    throw ConversionError(token, "Conversion error in bitwise & operation.");
  }

  k_value do_bitwise_or(const Token& token, const k_value& left,
                        const k_value& right) {
    if (std::holds_alternative<k_int>(left)) {
      auto lhs = std::get<k_int>(left);
      if (std::holds_alternative<k_int>(right)) {
        return lhs | std::get<k_int>(right);
      } else if (std::holds_alternative<double>(right)) {
        return lhs | static_cast<k_int>(std::get<double>(right));
      } else if (std::holds_alternative<bool>(right)) {
        k_int rhs = std::get<bool>(right) ? 1 : 0;
        return lhs | rhs;
      }
    }

    throw ConversionError(token, "Conversion error in bitwise | operation.");
  }

  k_value do_bitwise_xor(const Token& token, const k_value& left,
                         const k_value& right) {
    if (std::holds_alternative<k_int>(left)) {
      auto lhs = std::get<k_int>(left);
      if (std::holds_alternative<k_int>(right)) {
        return lhs ^ std::get<k_int>(right);
      } else if (std::holds_alternative<double>(right)) {
        return lhs ^ static_cast<k_int>(std::get<double>(right));
      } else if (std::holds_alternative<bool>(right)) {
        k_int rhs = std::get<bool>(right) ? 1 : 0;
        return lhs ^ rhs;
      }
    }

    throw ConversionError(token, "Conversion error in bitwise ^ operation.");
  }

  k_value do_bitwise_not(const Token& token, const k_value& left) {
    if (std::holds_alternative<k_int>(left)) {
      return ~std::get<k_int>(left);
    } else if (std::holds_alternative<double>(left)) {
      return ~static_cast<k_int>(std::get<double>(left));
    } else if (std::holds_alternative<bool>(left)) {
      return ~static_cast<k_int>(std::get<bool>(left) ? 1 : 0);
    }

    throw ConversionError(token, "Conversion error in bitwise ~ operation.");
  }

  k_value do_bitwise_lshift(const Token& token, const k_value& left,
                            const k_value& right) {
    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      return std::get<k_int>(left) << std::get<k_int>(right);
    }

    throw ConversionError(token, "Conversion error in bitwise << operation.");
  }

  k_value do_bitwise_rshift(const Token& token, const k_value& left,
                            const k_value& right) {
    if (std::holds_alternative<k_int>(left) &&
        std::holds_alternative<k_int>(right)) {
      return std::get<k_int>(left) >> std::get<k_int>(right);
    }

    throw ConversionError(token, "Conversion error in bitwise >> operation.");
  }

  k_value do_negation(const Token& token, const k_value& right) {
    if (std::holds_alternative<k_int>(right)) {
      return -std::get<k_int>(right);
    } else if (std::holds_alternative<double>(right)) {
      return -std::get<double>(right);
    } else {
      throw ConversionError(token,
                            "Unary minus applied to a non-numeric value.");
    }
  }

  k_value do_logical_not(const k_value& right) {
    if (std::holds_alternative<bool>(right)) {
      return !std::get<bool>(right);
    } else if (std::holds_alternative<k_null>(right)) {
      return true;
    } else if (std::holds_alternative<k_int>(right)) {
      return static_cast<k_int>(std::get<k_int>(right) == 0 ? 1 : 0);
    } else if (std::holds_alternative<double>(right)) {
      return std::get<double>(right) == 0;
    } else if (std::holds_alternative<k_string>(right)) {
      return std::get<k_string>(right).empty();
    } else if (std::holds_alternative<k_list>(right)) {
      return std::get<k_list>(right)->elements.empty();
    } else if (std::holds_alternative<k_hash>(right)) {
      return std::get<k_hash>(right)->keys.empty();
    } else {
      return false;  // Object, Lambda, etc.
    }
  }

  double get_double(const Token& token, const k_value& value) {
    if (std::holds_alternative<k_int>(value)) {
      return static_cast<double>(std::get<k_int>(value));
    } else if (std::holds_alternative<double>(value)) {
      return std::get<double>(value);
    }

    throw ConversionError(token, "Cannot convert value to a double value.");
  }

  double __epsilon__() { return std::numeric_limits<double>::epsilon(); }

  k_value __sin__(const Token& token, const k_value& value) {
    return sin(get_double(token, value));
  }

  k_value __sinh__(const Token& token, const k_value& value) {
    return sinh(get_double(token, value));
  }

  k_value __asin__(const Token& token, const k_value& value) {
    return asin(get_double(token, value));
  }

  k_value __tan__(const Token& token, const k_value& value) {
    return tan(get_double(token, value));
  }

  k_value __tanh__(const Token& token, const k_value& value) {
    return tanh(get_double(token, value));
  }

  k_value __atan__(const Token& token, const k_value& value) {
    return atan(get_double(token, value));
  }

  k_value __atan2__(const Token& token, const k_value& valueY,
                    const k_value& valueX) {
    return atan2(get_double(token, valueY), get_double(token, valueX));
  }

  k_value __cos__(const Token& token, const k_value& value) {
    return cos(get_double(token, value));
  }

  k_value __acos__(const Token& token, const k_value& value) {
    return acos(get_double(token, value));
  }

  k_value __cosh__(const Token& token, const k_value& value) {
    return cosh(get_double(token, value));
  }

  k_value __log__(const Token& token, const k_value& value) {
    return log(get_double(token, value));
  }

  k_value __log2__(const Token& token, const k_value& value) {
    return log2(get_double(token, value));
  }

  k_value __log10__(const Token& token, const k_value& value) {
    return log10(get_double(token, value));
  }

  k_value __log1p__(const Token& token, const k_value& value) {
    return log1p(get_double(token, value));
  }

  k_value __sqrt__(const Token& token, const k_value& value) {
    return sqrt(get_double(token, value));
  }

  k_value __cbrt__(const Token& token, const k_value& value) {
    return cbrt(get_double(token, value));
  }

  k_value __fmod__(const Token& token, const k_value& valueX,
                   const k_value& valueY) {
    return fmod(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __hypot__(const Token& token, const k_value& valueX,
                    const k_value& valueY) {
    return hypot(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __isfinite__(const Token& token, const k_value& value) {
    return std::isfinite(get_double(token, value));
  }

  k_value __isinf__(const Token& token, const k_value& value) {
    return std::isinf(get_double(token, value));
  }

  k_value __isnan__(const Token& token, const k_value& value) {
    return std::isnan(get_double(token, value));
  }

  k_value __isnormal__(const Token& token, const k_value& value) {
    return std::isnormal(get_double(token, value));
  }

  k_value __floor__(const Token& token, const k_value& value) {
    return floor(get_double(token, value));
  }

  k_value __ceil__(const Token& token, const k_value& value) {
    return ceil(get_double(token, value));
  }

  k_value __round__(const Token& token, const k_value& value) {
    return round(get_double(token, value));
  }

  k_value __trunc__(const Token& token, const k_value& value) {
    return trunc(get_double(token, value));
  }

  k_value __remainder__(const Token& token, const k_value& valueX,
                        const k_value& valueY) {
    return remainder(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __exp__(const Token& token, const k_value& value) {
    return exp(get_double(token, value));
  }

  k_value __expm1__(const Token& token, const k_value& value) {
    return expm1(get_double(token, value));
  }

  k_value __erf__(const Token& token, const k_value& value) {
    return erf(get_double(token, value));
  }

  k_value __erfc__(const Token& token, const k_value& value) {
    return erfc(get_double(token, value));
  }

  k_value __lgamma__(const Token& token, const k_value& value) {
    return lgamma(get_double(token, value));
  }

  k_value __tgamma__(const Token& token, const k_value& value) {
    return tgamma(get_double(token, value));
  }

  k_value __fdim__(const Token& token, const k_value& valueX,
                   const k_value& valueY) {
    return fdim(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __copysign__(const Token& token, const k_value& valueX,
                       const k_value& valueY) {
    return copysign(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __nextafter__(const Token& token, const k_value& valueX,
                        const k_value& valueY) {
    return nextafter(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __max__(const Token& token, const k_value& valueX,
                  const k_value& valueY) {
    return fmax(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __min__(const Token& token, const k_value& valueX,
                  const k_value& valueY) {
    return fmin(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __pow__(const Token& token, const k_value& valueX,
                  const k_value& valueY) {
    return pow(get_double(token, valueX), get_double(token, valueY));
  }

  k_value __rotr__(k_int value, k_int shift) {
    if (shift == 0) {
      return value;
    }

    auto unsignedValue = static_cast<unsigned long long>(value);
    const int bits = sizeof(unsignedValue) * CHAR_BIT;

    shift %= bits;

    if (shift < 0) {
      shift += bits;
    }

    unsigned long long result =
        (unsignedValue >> shift) | (unsignedValue << (bits - shift));

    return static_cast<k_int>(result);
  }

  k_value __rotl__(k_int value, k_int shift) {
    if (shift == 0) {
      return value;
    }

    unsigned long long unsignedValue = static_cast<unsigned long long>(value);
    const int bits = sizeof(unsignedValue) * CHAR_BIT;

    shift %= bits;

    if (shift < 0) {
      shift += bits;
    }

    unsigned long long result =
        (unsignedValue << shift) | (unsignedValue >> (bits - shift));

    return static_cast<k_int>(result);
  }

  k_value __abs__(const Token& token, const k_value& value) {
    if (std::holds_alternative<k_int>(value)) {
      return static_cast<k_int>(
          labs(static_cast<long>(std::get<k_int>(value))));
    } else if (std::holds_alternative<double>(value)) {
      return fabs(std::get<double>(value));
    }

    throw ConversionError(
        token, "Cannot take an absolute value of a non-numeric value.");
  }

  std::vector<k_value> __divisors__(int number) {
    std::vector<k_value> divisors;

    for (int i = 1; i <= sqrt(number); ++i) {
      if (number % i == 0) {
        divisors.emplace_back(static_cast<k_int>(i));
        if (i != number / i) {
          divisors.emplace_back(static_cast<k_int>(number / i));
        }
      }
    }

    return divisors;
  }

  k_value __random__(const Token& token, const k_value& valueX,
                     const k_value& valueY) {
    if (std::holds_alternative<k_string>(valueX)) {
      auto limit = get_integer(token, valueY);
      return RNG::getInstance().randomString(std::get<k_string>(valueX), limit);
    }

    if (std::holds_alternative<k_list>(valueX)) {
      auto limit = get_integer(token, valueY);
      return RNG::getInstance().randomList(std::get<k_list>(valueX), limit);
    }

    if (std::holds_alternative<double>(valueX) ||
        std::holds_alternative<double>(valueY)) {
      double x = get_double(token, valueX), y = get_double(token, valueY);
      return RNG::getInstance().random(x, y);
    } else if (std::holds_alternative<k_int>(valueX) ||
               std::holds_alternative<k_int>(valueY)) {
      auto x = get_integer(token, valueX), y = get_integer(token, valueY);
      return RNG::getInstance().random(x, y);
    }

    throw ConversionError(token,
                          "Expected a numeric value in random number range");
  }

  k_value do_unary_op(const Token& token, const KName& op,
                      const k_value& right) {
    switch (op) {
      case KName::Ops_Not:
        return do_logical_not(right);

      case KName::Ops_BitwiseNot:
      case KName::Ops_BitwiseNotAssign:
        return do_bitwise_not(token, right);

      case KName::Ops_Subtract:
        return do_negation(token, right);

      default:
        throw InvalidOperationError(token, "Unknown unary operation.");
    }
  }

  k_value do_binary_op(const Token& token, const KName& op, const k_value& left,
                       const k_value& right) {
    switch (op) {
      case KName::Ops_Add:
      case KName::Ops_AddAssign:
        return do_addition(token, left, right);
      case KName::Ops_Subtract:
      case KName::Ops_SubtractAssign:
        return do_subtraction(token, left, right);
      case KName::Ops_Multiply:
      case KName::Ops_MultiplyAssign:
        return do_multiplication(token, left, right);
      case KName::Ops_Divide:
      case KName::Ops_DivideAssign:
        return do_division(token, left, right);
      case KName::Ops_Modulus:
      case KName::Ops_ModuloAssign:
        return do_modulus(token, left, right);
      case KName::Ops_Exponent:
      case KName::Ops_ExponentAssign:
        return do_exponentiation(token, left, right);
      case KName::Ops_BitwiseAnd:
      case KName::Ops_BitwiseAndAssign:
        return do_bitwise_and(token, left, right);
      case KName::Ops_BitwiseOr:
      case KName::Ops_BitwiseOrAssign:
        return do_bitwise_or(token, left, right);
      case KName::Ops_BitwiseXor:
      case KName::Ops_BitwiseXorAssign:
        return do_bitwise_xor(token, left, right);
      case KName::Ops_BitwiseLeftShift:
      case KName::Ops_BitwiseLeftShiftAssign:
        return do_bitwise_lshift(token, left, right);
      case KName::Ops_BitwiseRightShift:
      case KName::Ops_BitwiseRightShiftAssign:
        return do_bitwise_rshift(token, left, right);
      case KName::Ops_And:
      case KName::Ops_AndAssign:
        return is_truthy(left) && is_truthy(right);
      case KName::Ops_Or:
      case KName::Ops_OrAssign:
        return is_truthy(left) || is_truthy(right);
      case KName::Ops_LessThan:
        return do_lt_comparison(left, right);
      case KName::Ops_LessThanOrEqual:
        return do_lte_comparison(left, right);
      case KName::Ops_GreaterThan:
        return do_gt_comparison(left, right);
      case KName::Ops_GreaterThanOrEqual:
        return do_gte_comparison(left, right);
      case KName::Ops_Equal:
        return do_eq_comparison(left, right);
      case KName::Ops_NotEqual:
        return do_neq_comparison(left, right);
      default:
        throw InvalidOperationError(token, "Unknown binary operation.");
    }
  }
} MathImpl;

struct {
  k_value __dropout__(const Token& token, const k_value& inputs,
                      const k_value& dropout_rate) {
    if (!std::holds_alternative<k_list>(inputs)) {
      throw ConversionError(
          token, "Expected a list of inputs for dropout regularization.");
    }

    const auto& dropoutRate = get_double(token, dropout_rate);
    auto& inputsList = std::get<k_list>(inputs)->elements;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& val : inputsList) {
      if (dis(gen) < dropoutRate) {
        val = 0.0;
      }
    }

    return inputs;
  }

  k_value __l1_regularization__(const Token& token, const k_value& weights,
                                const k_value& lambda) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(
          token, "Expected a list for weights in L1 regularization.");
    }

    if (!std::holds_alternative<double>(lambda)) {
      throw ConversionError(
          token, "Expected a double for lambda in L1 regularization.");
    }

    double sum = 0.0;
    double lambdaValue = get_double(token, lambda);
    const auto& weightValues = std::get<k_list>(weights)->elements;

    for (const auto& weight : weightValues) {
      sum += std::abs(get_double(token, weight));
    }
    return lambdaValue * sum;
  }

  k_value __l2_regularization__(const Token& token, const k_value& weights,
                                const k_value& lambda) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(
          token, "Expected a list for weights in L2 regularization.");
    }

    if (!std::holds_alternative<double>(lambda)) {
      throw ConversionError(
          token, "Expected a double for lambda in L2 regularization.");
    }

    double sum = 0.0;
    double lambdaValue = get_double(token, lambda);
    const auto& weightValues = std::get<k_list>(weights)->elements;

    for (const auto& weight : weightValues) {
      auto w = get_double(token, weight);
      sum += w * w;
    }

    return lambdaValue * sum;
  }

  k_value __elastic_net__(const Token& token, const k_value& weights,
                          const k_value& lambda1 = 0.01,
                          const k_value& lambda2 = 0.01) {
    const auto& l1_term =
        get_double(token, __l1_regularization__(token, weights, lambda1));
    const auto& l2_term =
        get_double(token, __l2_regularization__(token, weights, lambda2));

    return l1_term + l2_term;
  }

  void __weight_decay__(const Token& token, const k_value& weights,
                        const k_value& lambda = 0.01) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(token, "Expected list in weight decay function.");
    }

    auto& weightsList = std::get<k_list>(weights)->elements;
    const auto& lambdaValue = get_double(token, lambda);
    for (auto& weight : weightsList) {
      const auto& w = get_double(token, weight);
      weight = w - (lambdaValue * w);
    }
  }
} MLRegularizationBuiltins;

struct {
  void __rmsprop__(const Token& token, k_value& weights,
                   const k_value& gradients, k_value& v,
                   const k_value& learning_rate, const k_value& decay_rate) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(token,
                            "Expected a list for weights in root mean squared "
                            "propagation optimizer.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(token,
                            "Expected a list for gradients in root mean "
                            "squared propagation optimizer.");
    }
    if (!std::holds_alternative<k_list>(v)) {
      throw ConversionError(
          token,
          "Expected a list for running average of squared gradients in root "
          "mean squared propagation optimizer.");
    }

    const auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& vList = std::get<k_list>(v)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != vList.size()) {
      throw InvalidOperationError(token,
                                  "All lists must be the same size in root "
                                  "mean squared propagation optimizer.");
    }

    const auto& learningRate = get_double(token, learning_rate);
    const auto& decayRate = get_double(token, decay_rate);

    for (size_t i = 0; i < weightsList.size(); ++i) {
      const auto& gradient = get_double(token, gradientsList[i]);
      vList[i] = decayRate * get_double(token, vList[i]) +
                 (1.0 - decayRate) * gradient * gradient;
      weightsList[i] =
          get_double(token, weightsList[i]) -
          learningRate * gradient /
              (std::sqrt(get_double(token, vList[i])) + MathImpl.__epsilon__());
    }
  }

  void __adadelta__(const Token& token, k_value& weights,
                    const k_value& gradients, k_value& accum_grad,
                    k_value& accum_update, const k_value& rho) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(
          token, "Expected a list for weights in adaptive gradient optimizer.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(
          token,
          "Expected a list for gradients in adaptive gradient optimizer.");
    }
    if (!std::holds_alternative<k_list>(accum_grad)) {
      throw ConversionError(token,
                            "Expected a list for accumulation of squared "
                            "gradients in adaptive gradient optimizer.");
    }
    if (!std::holds_alternative<k_list>(accum_update)) {
      throw ConversionError(token,
                            "Expected a list for accumulation of squared "
                            "updates in adaptive gradient optimizer.");
    }

    const auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& accumGradList = std::get<k_list>(accum_grad)->elements;
    auto& accumUpdateList = std::get<k_list>(accum_update)->elements;
    auto epsilon = MathImpl.__epsilon__();
    const auto& rhoValue = get_double(token, rho);

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != accumGradList.size() ||
        weightsList.size() != accumUpdateList.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in adaptive gradient optimizer.");
    }

    for (size_t i = 0; i < weightsList.size(); ++i) {
      const auto& grad = get_double(token, gradientsList[i]);
      accumGradList[i] = rhoValue * get_double(token, accumGradList[i]) +
                         (1 - rhoValue) * grad * grad;
      const auto& accumUpdate = get_double(token, accumUpdateList[i]);
      double update =
          std::sqrt((accumUpdate + epsilon) /
                    (get_double(token, accumGradList[i]) + epsilon)) *
          grad;

      accumUpdateList[i] =
          rhoValue * accumUpdate + (1 - rhoValue) * update * update;
      weightsList[i] = get_double(token, weightsList[i]) - update;
    }
  }

  void __adagrad__(const Token& token, k_value& weights,
                   const k_value& gradients, k_value& v,
                   const k_value& learning_rate = 0.01) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(
          token, "Expected a list for weights in adaptive gradient optimizer.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(
          token,
          "Expected a list for gradients in adaptive gradient optimizer.");
    }
    if (!std::holds_alternative<k_list>(v)) {
      throw ConversionError(token,
                            "Expected a list for sum of squared gradients in "
                            "adaptive gradient optimizer.");
    }

    const auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& vList = std::get<k_list>(v)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != vList.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in adaptive gradient optimizer.");
    }

    const auto& learningRate = get_double(token, learning_rate);

    for (size_t i = 0; i < weightsList.size(); ++i) {
      const auto& gradient = get_double(token, gradientsList[i]);
      vList[i] = get_double(token, vList[i]) + (gradient * gradient);
      weightsList[i] =
          get_double(token, weightsList[i]) -
          (learningRate * gradient /
           (std::sqrt(get_double(token, vList[i])) + MathImpl.__epsilon__()));
    }
  }

  void __adamax__(const Token& token, k_value& weights,
                  const k_value& gradients, k_value& m, k_value& v,
                  const k_value& learning_rate, const k_value& beta1,
                  const k_value& beta2) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(token,
                            "Expected a list for weights in adaptive moment "
                            "estimation (max norm) optimizer.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(token,
                            "Expected a list for gradients in adaptive moment "
                            "estimation (max norm) optimizer.");
    }
    if (!std::holds_alternative<k_list>(m)) {
      throw ConversionError(token,
                            "Expected a list for first moment estimate in "
                            "adaptive moment estimation (max norm) optimizer.");
    }
    if (!std::holds_alternative<k_list>(v)) {
      throw ConversionError(token,
                            "Expected a list for second moment estimate in "
                            "adaptive moment estimation (max norm) optimizer.");
    }

    const auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& mList = std::get<k_list>(m)->elements;
    auto& vList = std::get<k_list>(v)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != mList.size() ||
        weightsList.size() != vList.size()) {
      throw InvalidOperationError(token,
                                  "All lists must be the same size in adaptive "
                                  "moment estimation (max norm) optimizer.");
    }

    const auto& learningRate = get_double(token, learning_rate);
    const auto& beta1Value = get_double(token, beta1);
    const auto& beta2Value = get_double(token, beta2);

    for (size_t i = 0; i < weightsList.size(); ++i) {
      const auto& grad = get_double(token, gradientsList[i]);
      mList[i] =
          beta1Value * get_double(token, mList[i]) + (1 - beta1Value) * grad;
      vList[i] =
          std::max(beta2Value * get_double(token, vList[i]), std::abs(grad));

      weightsList[i] = get_double(token, weightsList[i]) -
                       (learningRate * get_double(token, mList[i]) /
                        (get_double(token, vList[i]) + MathImpl.__epsilon__()));
    }
  }

  void __adam__(const Token& token, k_value& weights, const k_value& gradients,
                k_value& m, k_value& v, const k_value& learning_rate,
                const k_value& beta1, const k_value& beta2, const k_value& t) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(token,
                            "Expected a list for weights in adaptive moment "
                            "estimation optimizer.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(token,
                            "Expected a list for gradients in adaptive moment "
                            "estimation optimizer.");
    }
    if (!std::holds_alternative<k_list>(m)) {
      throw ConversionError(token,
                            "Expected a list for first moment estimate in "
                            "adaptive moment estimation optimizer.");
    }
    if (!std::holds_alternative<k_list>(v)) {
      throw ConversionError(token,
                            "Expected a list for second moment estimate in "
                            "adaptive moment estimation optimizer.");
    }

    const auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& mList = std::get<k_list>(m)->elements;
    auto& vList = std::get<k_list>(v)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != mList.size() ||
        weightsList.size() != vList.size()) {
      throw InvalidOperationError(token,
                                  "All lists must be the same size in adaptive "
                                  "moment estimation optimizer.");
    }

    const auto& learningRate = get_double(token, learning_rate);
    const auto& beta1Value = get_double(token, beta1);
    const auto& beta2Value = get_double(token, beta2);
    const auto& tValue = get_integer(token, t);

    for (size_t i = 0; i < weightsList.size(); ++i) {
      const auto& gradient = get_double(token, gradientsList[i]);
      mList[i] = beta1Value * get_double(token, mList[i]) +
                 (1.0 - beta1Value) * gradient;
      vList[i] = beta2Value * get_double(token, vList[i]) +
                 (1.0 - beta2Value) * gradient * gradient;

      double m_hat =
          get_double(token, mList[i]) / (1.0 - std::pow(beta1Value, tValue));
      double v_hat =
          get_double(token, vList[i]) / (1.0 - std::pow(beta2Value, tValue));

      weightsList[i] =
          get_double(token, weightsList[i]) -
          learningRate * m_hat / (std::sqrt(v_hat) + MathImpl.__epsilon__());
    }
  }

  void __nadam__(const Token& token, k_value& weights, const k_value& gradients,
                 k_value& m, k_value& v, const k_value& learning_rate,
                 const k_value& beta1, const k_value& beta2, const k_value& t) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(
          token,
          "Expected a list for weights in Nesterov-accelerated adaptive moment "
          "estimation optimizer.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(
          token,
          "Expected a list for gradients in Nesterov-accelerated adaptive "
          "moment estimation optimizer.");
    }
    if (!std::holds_alternative<k_list>(m)) {
      throw ConversionError(
          token,
          "Expected a list for first moment estimate in Nesterov-accelerated "
          "adaptive moment estimation optimizer.");
    }
    if (!std::holds_alternative<k_list>(v)) {
      throw ConversionError(
          token,
          "Expected a list for second moment estimate in Nesterov-accelerated "
          "adaptive moment estimation optimizer.");
    }

    const auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& mList = std::get<k_list>(m)->elements;
    auto& vList = std::get<k_list>(v)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != mList.size() ||
        weightsList.size() != vList.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in Nesterov-accelerated adaptive "
          "moment estimation optimizer.");
    }

    const auto& learningRate = get_double(token, learning_rate);
    const auto& beta1Value = get_double(token, beta1);
    const auto& beta2Value = get_double(token, beta2);
    const auto& tValue = get_integer(token, t);
    double beta1_t = beta1Value * (1 - std::pow(0.1, tValue / 1000.0));

    for (size_t i = 0; i < weightsList.size(); ++i) {
      const auto& grad = get_double(token, gradientsList[i]);
      mList[i] =
          beta1Value * get_double(token, mList[i]) + (1 - beta1Value) * grad;
      vList[i] = beta2Value * get_double(token, vList[i]) +
                 (1 - beta2Value) * grad * grad;

      double m_hat = get_double(token, mList[i]) / (1 - beta1_t);
      double v_hat = get_double(token, vList[i]) / (1 - beta2Value);

      weightsList[i] =
          get_double(token, weightsList[i]) -
          (learningRate * (beta1Value * m_hat + (1 - beta1Value) * grad) /
           (std::sqrt(v_hat) + MathImpl.__epsilon__()));
    }
  }

  void __sgd__(const Token& token, k_value& weights, const k_value& gradients,
               k_value& velocity, const k_value& learning_rate,
               const k_value& momentum) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(
          token, "Expected a list for weights in stochastic gradient descent.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(
          token,
          "Expected a list for gradients in stochastic gradient descent.");
    }
    if (!std::holds_alternative<k_list>(velocity)) {
      throw ConversionError(
          token,
          "Expected a list for velocity in stochastic gradient descent.");
    }
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& velocityList = std::get<k_list>(velocity)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != velocityList.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in stochastic gradient descent.");
    }

    const auto& momentumValue = get_double(token, momentum);
    const auto& learningRate = get_double(token, learning_rate);
    for (size_t i = 0; i < weightsList.size(); ++i) {
      velocityList[i] = momentumValue * get_double(token, velocityList[i]) -
                        learningRate * get_double(token, gradientsList[i]);
      weightsList[i] =
          MathImpl.do_addition(token, weightsList[i], velocityList[i]);
    }
  }

  void __nesterov_sgd__(const Token& token, k_value& weights,
                        const k_value& gradients, k_value& velocity,
                        const k_value& learning_rate, const k_value& momentum) {
    if (!std::holds_alternative<k_list>(weights)) {
      throw ConversionError(token,
                            "Expected a list for weights in Nesterov "
                            "stochastic gradient descent.");
    }
    if (!std::holds_alternative<k_list>(gradients)) {
      throw ConversionError(token,
                            "Expected a list for gradients in Nesterov "
                            "stochastic gradient descent.");
    }
    if (!std::holds_alternative<k_list>(velocity)) {
      throw ConversionError(token,
                            "Expected a list for velocity in Nesterov "
                            "stochastic gradient descent.");
    }
    auto& weightsList = std::get<k_list>(weights)->elements;
    auto& gradientsList = std::get<k_list>(gradients)->elements;
    auto& velocityList = std::get<k_list>(velocity)->elements;

    if (weightsList.size() != gradientsList.size() ||
        weightsList.size() != velocityList.size()) {
      throw InvalidOperationError(token,
                                  "All lists must be the same size in Nesterov "
                                  "stochastic gradient descent.");
    }

    const auto& momentumValue = get_double(token, momentum);
    const auto& learningRate = get_double(token, learning_rate);

    for (size_t i = 0; i < weightsList.size(); ++i) {
      double lookahead_weight =
          get_double(token, weightsList[i]) +
          momentumValue * get_double(token, velocityList[i]);
      velocityList[i] = momentumValue * get_double(token, velocityList[i]) -
                        learningRate * get_double(token, gradientsList[i]);
      weightsList[i] = lookahead_weight + get_double(token, velocityList[i]);
    }
  }
} MLOptimizerBuiltins;

struct {
  k_value __binary_crossentropy__(const Token& token, const k_value& y_true,
                                  const k_value& y_pred) {
    const auto& yTrue = get_double(token, y_true);
    const auto& yPred = get_double(token, y_pred);

    return -(yTrue * std::log(yPred) + (1 - yTrue) * std::log(1 - yPred));
  }

  double __binary_focal_loss__(const Token& token, const k_value& y_true,
                               const k_value& y_pred,
                               const k_value& gamma = 2.0,
                               const k_value& alpha = 0.25) {
    double epsilon = MathImpl.__epsilon__();
    double yPred = get_double(token, y_pred);
    double gammaValue = get_double(token, gamma);
    double alphaValue = get_double(token, alpha);
    yPred = std::max(std::min(yPred, 1.0 - epsilon), epsilon);

    if (get_double(token, y_true) == 1.0) {
      return -alphaValue * std::pow(1.0 - yPred, gammaValue) * std::log(yPred);
    }

    return -(1.0 - alphaValue) * std::pow(yPred, gammaValue) *
           std::log(1.0 - yPred);
  }

  k_value __categorical_crossentropy__(const Token& token,
                                       const k_value& y_true,
                                       const k_value& y_pred) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(token,
                            "Expected a list for actual values in categorical "
                            "cross entropy function.");
    }
    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(token,
                            "Expected a list for predicted values in "
                            "categorical cross entropy function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(token,
                                  "All lists must be the same size in "
                                  "categorical cross entropy function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(
          token,
          "Expected non-empty lists in categorical cross entropy function.");
    }

    double loss = 0.0;
    double pred = 0.0;

    for (size_t i = 0; i < yTrue.size(); ++i) {
      pred = std::max(get_double(token, yPred[i]), MathImpl.__epsilon__());
      loss -= get_double(token, yTrue[i]) * std::log(pred);
    }

    return loss;
  }

  k_value __cosine_similarity__(const Token& token, const k_value& y_true,
                                const k_value& y_pred) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(
          token,
          "Expected a list for actual values in cosine similarity function.");
    }
    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(token,
                            "Expected a list for predicted values in cosine "
                            "similarity function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in cosine similarity function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(
          token, "Expected non-empty lists in cosine similarity function.");
    }

    double dot_product = 0.0;
    double norm_true = 0.0;
    double norm_pred = 0.0;

    for (size_t i = 0; i < yTrue.size(); ++i) {
      const auto& yTrueValue = get_double(token, yTrue[i]);
      const auto& yPredValue = get_double(token, yPred[i]);
      dot_product += yTrueValue * yPredValue;
      norm_true += yTrueValue * yTrueValue;
      norm_pred += yPredValue * yPredValue;
    }

    norm_true = std::sqrt(norm_true);
    norm_pred = std::sqrt(norm_pred);

    if (norm_true == 0) {
      throw InvalidOperationError(
          token, "The list of actual values is a zero vector.");
    }

    if (norm_pred == 0) {
      throw InvalidOperationError(
          token, "The list of predicted values is a zero vector.");
    }

    return dot_product / (norm_true * norm_pred);
  }

  k_value __dice_loss__(const Token& token, const k_value& y_true,
                        const k_value& y_pred) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(
          token, "Expected a list for actual values in Dice loss function.");
    }
    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(
          token, "Expected a list for predicted values in Dice loss function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(
          token, "All lists must be the same size in Dice loss function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(token,
                           "Expected non-empty lists in Dice loss function.");
    }

    double intersection = 0.0;
    double union_sum = 0.0;
    double epsilon = MathImpl.__epsilon__();

    for (size_t i = 0; i < yTrue.size(); ++i) {
      const auto& yTrueValue = get_double(token, yTrue[i]);
      const auto& yPredValue = get_double(token, yPred[i]);
      intersection += yTrueValue * yPredValue;
      union_sum += yTrueValue + yPredValue;
    }

    double dice = (2.0 * intersection + epsilon) / (union_sum + epsilon);

    return 1.0 - dice;
  }

  k_value __focal_loss__(const Token& token, const k_value& y_true,
                         const k_value& y_pred, const k_value& gamma = 2.0,
                         const k_value& alpha = 0.25) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(
          token, "Expected a list for actual values in focal loss function.");
    }
    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(
          token,
          "Expected a list for predicted values in focal loss function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(
          token, "All lists must be the same size in focal loss function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(token,
                           "Expected non-empty lists in focal loss function.");
    }

    double epsilon = MathImpl.__epsilon__();
    double loss = 0.0;
    const auto& alphaValue = get_double(token, alpha);
    const auto& gammaValue = get_double(token, gamma);

    for (size_t i = 0; i < yTrue.size(); ++i) {
      const auto& yTrueValue = get_double(token, yTrue[i]);
      const auto& yPredValue = get_double(token, yPred[i]);
      double pred = std::max(std::min(yPredValue, 1.0 - epsilon), epsilon);
      double p_t = yTrueValue ? pred : (1 - pred);
      double focal_weight = yTrueValue
                                ? alphaValue * std::pow(1 - pred, gammaValue)
                                : (1 - alphaValue) * std::pow(pred, gammaValue);
      loss -= focal_weight * std::log(p_t);
    }

    return loss;
  }

  k_value __kldivergence__(const Token& token, const k_value& y_true,
                           const k_value& y_pred) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(token,
                            "Expected a list for actual values in "
                            "Kullback-Leibler divergence function.");
    }
    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(token,
                            "Expected a list for predicted values in "
                            "Kullback-Leibler divergence function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(token,
                                  "All lists must be the same size in "
                                  "Kullback-Leibler divergence function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(
          token,
          "Expected non-empty lists in Kullback-Leibler divergence function.");
    }

    double epsilon = MathImpl.__epsilon__();

    double kl_div = 0.0;
    size_t size = yTrue.size();
    for (size_t i = 0; i < size; ++i) {
      double p = std::max(get_double(token, yTrue[i]), epsilon);
      double q = std::max(get_double(token, yPred[i]), epsilon);
      kl_div += p * std::log(p / q);
    }
    return kl_div;
  }

  k_value __hinge_loss__(const Token& token, const k_value& y_true,
                         const k_value& y_pred) {
    const auto& yTrue = get_double(token, y_true);
    const auto& yPred = get_double(token, y_pred);

    return std::max(0.0, 1.0 - yTrue * yPred);
  }

  k_value __huber_loss__(const Token& token, const k_value& y_true,
                         const k_value& y_pred, const k_value& delta) {
    const auto& yTrue = get_double(token, y_true);
    const auto& yPred = get_double(token, y_pred);
    const auto& d = get_double(token, delta);
    double diff = yTrue - yPred;

    if (std::abs(diff) <= d) {
      return 0.5 * diff * diff;
    }

    return d * (std::abs(diff) - 0.5 * d);
  }

  k_value __log_cosh__(const Token& token, const k_value& y_true,
                       const k_value& y_pred) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(
          token,
          "Expected a list for actual values in mean squared error function.");
    }

    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(token,
                            "Expected a list for predicted values in mean "
                            "squared error function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in mean squared error function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(
          token, "Expected non-empty lists in mean squared error function.");
    }

    double loss = 0.0;
    size_t size = yTrue.size();

    for (size_t i = 0; i < size; ++i) {
      double error = get_double(token, yTrue[i]) - get_double(token, yPred[i]);
      loss += std::log(std::cosh(error));
    }

    return loss / size;
  }

  k_value __mae__(const Token& token, const k_value& y_true,
                  const k_value& y_pred) {
    const auto& yTrue = get_double(token, y_true);
    const auto& yPred = get_double(token, y_pred);

    return std::abs(yTrue - yPred);
  }

  k_value __mse__(const Token& token, const k_value& y_true,
                  const k_value& y_pred) {
    if (!std::holds_alternative<k_list>(y_true)) {
      throw ConversionError(
          token,
          "Expected a list for actual values in mean squared error function.");
    }
    if (!std::holds_alternative<k_list>(y_pred)) {
      throw ConversionError(token,
                            "Expected a list for predicted values in mean "
                            "squared error function.");
    }

    const auto& yTrue = std::get<k_list>(y_true)->elements;
    const auto& yPred = std::get<k_list>(y_pred)->elements;

    if (yTrue.size() != yPred.size()) {
      throw InvalidOperationError(
          token,
          "All lists must be the same size in mean squared error function.");
    }

    if (yTrue.empty()) {
      throw EmptyListError(
          token, "Expected non-empty lists in mean squared error function.");
    }

    double sum = 0.0;

    for (size_t i = 0; i < yTrue.size(); ++i) {
      sum += std::pow(get_double(token, yTrue[i]) - get_double(token, yPred[i]),
                      2);
    }

    return sum / yTrue.size();
  }

  k_value __quantile_loss__(const Token& token, const k_value& y_true,
                            const k_value& y_pred,
                            const k_value& quantile = 0.5) {
    double diff = get_double(token, y_true) - get_double(token, y_pred);
    double q = get_double(token, quantile);
    if (diff > 0) {
      return q * diff;
    }

    return (1.0 - q) * (-diff);
  }
} MLLossBuiltins;

struct {
  k_value __elu__(const Token& token, const k_value& x, const k_value& alpha) {
    auto xValue = get_double(token, x);
    auto alphaValue = get_double(token, alpha);

    return (xValue > 0) ? xValue : alphaValue * (std::exp(xValue) - 1);
  }

  k_value __gelu__(const Token& token, const k_value& xValue) {
    const double& x = get_double(token, xValue);
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    const double coeff = 0.044715;

    return 0.5 * x *
           (1.0 + std::tanh(sqrt_2_over_pi * (x + coeff * std::pow(x, 3))));
  }

  k_value __gelu_approx__(const Token& token, const k_value& xValue) {
    const double& x = get_double(token, xValue);
    const double sqrt2_pi = std::sqrt(2.0 / M_PI);
    return 0.5 * x *
           (1.0 + std::tanh(sqrt2_pi * (x + 0.044715 * std::pow(x, 3))));
  }

  k_value __relu__(const Token& token, const k_value& x) {
    return std::max(0.0, get_double(token, x));
  }

  k_value __prelu__(const Token& token, const k_value& x,
                    const k_value& alpha) {
    if (!std::holds_alternative<k_list>(x)) {
      throw ConversionError(
          token, "Expected list of inputs in parametric ReLU function.");
    }

    const auto& xList = std::get<k_list>(x)->elements;
    const auto& alphaValue = get_double(token, alpha);

    std::vector<k_value> result(xList.size());
    for (size_t i = 0; i < xList.size(); ++i) {
      const auto& xValue = get_double(token, xList[i]);
      result[i] = xValue > 0 ? xValue : alphaValue * xValue;
    }
    return std::make_shared<List>(result);
  }

  k_value __sigmoid__(const Token& token, const k_value& x) {
    return 1.0 / (1.0 + std::exp(-get_double(token, x)));
  }

  k_value __softmax__(const Token& token, const k_value& inputs) {
    std::vector<double> exp_values;
    std::vector<k_value> probs;

    if (!std::holds_alternative<k_list>(inputs)) {
      throw ConversionError(token, "Expected a list for softmax.");
    }

    const auto& elements = std::get<k_list>(inputs)->elements;

    if (elements.empty()) {
      return inputs;
    }

    exp_values.reserve(elements.size());

    for (const auto& val : elements) {
      exp_values.push_back(std::exp(get_double(token, val)));
    }

    double sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);

    probs.reserve(exp_values.size());

    for (double exp_value : exp_values) {
      probs.emplace_back(exp_value / sum_exp);
    }

    return std::make_shared<List>(probs);
  }

  k_value __softplus__(const Token& token, const k_value& x) {
    return std::log(1 + std::exp(get_double(token, x)));
  }

  k_value softsign(const Token& token, const k_value& x) {
    if (!std::holds_alternative<k_list>(x)) {
      throw ConversionError(token, "Expected list in softsign function.");
    }

    const auto& xList = std::get<k_list>(x)->elements;

    std::vector<k_value> result;
    result.reserve(xList.size());
    for (const auto& value : xList) {
      const auto& d = get_double(token, value);
      result.push_back(d / (1 + std::fabs(d)));
    }

    return std::make_shared<List>(result);
  }

  k_value __selu__(const Token& token, const k_value& x) {
    const auto& xValue = get_double(token, x);
    const double lambda = 1.0507;
    const double alpha = 1.67326;

    return (xValue > 0) ? lambda * xValue
                        : lambda * alpha * (std::exp(xValue) - 1);
  }

  k_value __swish__(const Token& token, const k_value& x, const k_value& beta) {
    const auto& xValue = get_double(token, x);
    const auto& betaValue = get_double(token, beta);
    return xValue * std::get<double>(__sigmoid__(token, betaValue * xValue));
  }

  k_value __tanh_activation__(const Token& token, const k_value& x) {
    if (!std::holds_alternative<k_list>(x)) {
      throw ConversionError(
          token, "Expected list of inputs in parametric ReLU function.");
    }

    const auto& xList = std::get<k_list>(x)->elements;
    std::vector<k_value> result(xList.size());

    for (size_t i = 0; i < xList.size(); ++i) {
      result[i] = std::tanh(get_double(token, xList[i]));
    }

    return std::make_shared<List>(result);
  }

  k_value tanh_shrink(const Token& token, const k_value& xValue) {
    const auto& x = get_double(token, xValue);
    return x - std::tanh(x);
  }

  k_value __leaky_relu__(const Token& token, const k_value& x,
                         const k_value& alpha) {
    const auto& xValue = get_double(token, x);
    const auto& alphaValue = get_double(token, alpha);

    return (xValue > 0) ? xValue : alphaValue * xValue;
  }

  k_value __linear__(const Token& token, const k_value& x) {
    return get_double(token, x);
  }
} MLActivationBuiltins;

#endif