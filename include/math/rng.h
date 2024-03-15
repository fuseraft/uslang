#ifndef KIWI_MATH_RNG_H
#define KIWI_MATH_RNG_H

#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include "typing/value.h"

class RNG {
 public:
  static RNG& getInstance();
  double random(double from, double to);
  k_int random(k_int from, k_int to);
  std::string random16();
  std::string randomString(const std::string& input, size_t length);
  Value randomList(std::shared_ptr<List> list, size_t length);

 private:
  RNG();
  std::mt19937 generator;  // Mersenne Twister engine
};

RNG& RNG::getInstance() {
  static RNG instance;
  return instance;
}

RNG::RNG() {
  std::seed_seq seed{static_cast<unsigned>(std::time(0))};
  generator.seed(seed);
}

double RNG::random(double from, double to) {
  std::uniform_real_distribution<double> distribution(from, to);
  return distribution(generator);
}

k_int RNG::random(k_int from, k_int to) {
  std::uniform_int_distribution<k_int> distribution(from, to);
  return distribution(generator);
}

std::string RNG::random16() {
  const size_t LENGTH = 16;
  // Might trim it down a bit.
  const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  return randomString(chars, LENGTH);
}

std::string RNG::randomString(const std::string& chars, size_t length) {
  if (chars.empty()) {
    return chars;
  }

  std::uniform_int_distribution<> distribution(0, chars.size() - 1);
  std::ostringstream randomString;

  for (size_t i = 0; i < length; ++i) {
    randomString << chars[distribution(generator)];
  }

  return randomString.str();
}

Value RNG::randomList(std::shared_ptr<List> list, size_t length) {
  const auto& elements = list->elements;
  if (elements.empty()) {
    return std::make_shared<List>();
  }

  std::uniform_int_distribution<> distribution(0, elements.size() - 1);
  auto randomList = std::make_shared<List>();

  for (size_t i = 0; i < length; ++i) {
    randomList->elements.push_back(elements.at(distribution(generator)));
  }

  return randomList;
}

#endif