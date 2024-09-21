#ifndef KIWI_BUILTINS_ML_H
#define KIWI_BUILTINS_ML_H

#include <cstdlib>
#include <string>
#include <unordered_map>
#include "math/functions.h"
#include "parsing/builtins.h"
#include "parsing/tokens.h"
#include "typing/value.h"
#include "util/string.h"

const double HALF = 0.5;
const double ONE_PERCENT = 0.01;

class MLBuiltinHandler {
 public:
  static k_value execute(const Token& term, const KName& builtin,
                         const std::vector<k_value>& args) {
    switch (builtin) {
      case KName::Builtin_MLReg_Dropout:
        return executeRegDropout(term, args);

      case KName::Builtin_MLReg_WeightDecay:
        return executeRegWeightDecay(term, args);

      case KName::Builtin_MLReg_L1Lasso:
        return executeRegL1Lasso(term, args);

      case KName::Builtin_MLReg_L2Ridge:
        return executeRegL2Ridge(term, args);

      case KName::Builtin_MLReg_ElasticNet:
        return executeRegElasticNet(term, args);

      case KName::Builtin_MLOptim_RMSProp:
        return executeOptimRMSProp(term, args);

      case KName::Builtin_MLOptim_Adadelta:
        return executeOptimAdadelta(term, args);

      case KName::Builtin_MLOptim_Adagrad:
        return executeOptimAdagrad(term, args);

      case KName::Builtin_MLOptim_Adamax:
        return executeOptimAdamax(term, args);

      case KName::Builtin_MLOptim_Adam:
        return executeOptimAdam(term, args);

      case KName::Builtin_MLOptim_Nadam:
        return executeOptimNadam(term, args);

      case KName::Builtin_MLOptim_SGD:
        return executeOptimSGD(term, args);

      case KName::Builtin_MLOptim_SGDNesterov:
        return executeOptimSGDNesterov(term, args);

      case KName::Builtin_MLLoss_BinaryCrossEntropy:
        return executeLossBinaryCrossEntropy(term, args);

      case KName::Builtin_MLLoss_BinaryFocal:
        return executeLossBinaryFocal(term, args);

      case KName::Builtin_MLLoss_CatCrossEntropy:
        return executeLossCatCrossEntropy(term, args);

      case KName::Builtin_MLLoss_CosSimilarity:
        return executeLossCatCrossEntropy(term, args);

      default:
        break;
    }

    throw UnknownBuiltinError(term, "");
  }

 private:
  static k_value executeLossCosineSimilarity(const Token& term,
                                             const std::vector<k_value>& args) {
    if (args.size() != 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.LossCosSimilarity);
    }

    return MLLossBuiltins.__cosine_similarity__(term, args.at(0), args.at(1));
  }

  static k_value executeLossCatCrossEntropy(const Token& term,
                                            const std::vector<k_value>& args) {
    if (args.size() < 2 || args.size() > 3) {
      throw BuiltinUnexpectedArgumentError(term,
                                           MLBuiltins.LossCatCrossEntropy);
    }

    k_value epsilon = MathImpl.__epsilon__();
    epsilon = args.size() > 2 ? args.at(2) : epsilon;
    return MLLossBuiltins.__categorical_crossentropy__(term, args.at(0),
                                                       args.at(1), epsilon);
  }

  static k_value executeLossBinaryFocal(const Token& term,
                                        const std::vector<k_value>& args) {
    if (args.size() < 2 || args.size() > 5) {
      throw BuiltinUnexpectedArgumentError(term,
                                           MLBuiltins.LossBinaryCrossEntropy);
    }

    k_value gamma = 2.0;
    k_value alpha = 0.25;
    k_value epsilon = MathImpl.__epsilon__();

    gamma = args.size() > 2 ? args.at(2) : gamma;
    alpha = args.size() > 3 ? args.at(3) : alpha;
    epsilon = args.size() > 4 ? args.at(4) : epsilon;

    return MLLossBuiltins.__binary_focal_loss__(term, args.at(0), args.at(1),
                                                gamma, alpha, epsilon);
  }

  static k_value executeLossBinaryCrossEntropy(
      const Token& term, const std::vector<k_value>& args) {
    if (args.size() < 2 || args.size() > 3) {
      throw BuiltinUnexpectedArgumentError(term,
                                           MLBuiltins.LossBinaryCrossEntropy);
    }

    k_value epsilon = MathImpl.__epsilon__();

    epsilon = args.size() > 2 ? args.at(2) : epsilon;

    return MLLossBuiltins.__binary_crossentropy__(term, args.at(0), args.at(1),
                                                  epsilon);
  }

  static k_value executeOptimSGDNesterov(const Token& term,
                                         const std::vector<k_value>& args) {
    if (args.size() < 3 || args.size() > 5) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimSGDNesterov);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto velocity = args.at(2);
    k_value learningRate = 0.01;
    k_value momentum = 0.0;

    learningRate = args.size() > 3 ? args.at(3) : learningRate;
    momentum = args.size() > 4 ? args.at(4) : momentum;

    MLOptimizerBuiltins.__nesterov_sgd__(term, weights, gradients, velocity,
                                         learningRate, momentum);
    return weights;
  }

  static k_value executeOptimSGD(const Token& term,
                                 const std::vector<k_value>& args) {
    if (args.size() < 3 || args.size() > 5) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimSGD);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto velocity = args.at(2);
    k_value learningRate = 0.01;
    k_value momentum = 0.0;

    learningRate = args.size() > 3 ? args.at(3) : learningRate;
    momentum = args.size() > 4 ? args.at(4) : momentum;

    MLOptimizerBuiltins.__sgd__(term, weights, gradients, velocity,
                                learningRate, momentum);
    return weights;
  }

  static k_value executeOptimNadam(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.size() < 4 || args.size() > 9) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimNadam);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto m = args.at(2);
    auto v = args.at(3);
    k_value learningRate = 0.02;
    k_value beta1 = 0.9;
    k_value beta2 = 0.999;
    k_value t = 1;
    k_value epsilon = MathImpl.__epsilon__();

    learningRate = args.size() > 4 ? args.at(4) : learningRate;
    beta1 = args.size() > 5 ? args.at(5) : beta1;
    beta2 = args.size() > 6 ? args.at(6) : beta2;
    t = args.size() > 7 ? args.at(7) : t;
    epsilon = args.size() > 8 ? args.at(8) : epsilon;

    MLOptimizerBuiltins.__nadam__(term, weights, gradients, m, v, learningRate,
                                  beta1, beta2, t, epsilon);
    return weights;
  }

  static k_value executeOptimAdam(const Token& term,
                                  const std::vector<k_value>& args) {
    if (args.size() < 4 || args.size() > 9) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimAdam);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto m = args.at(2);
    auto v = args.at(3);
    k_value learningRate = 0.02;
    k_value beta1 = 0.9;
    k_value beta2 = 0.999;
    k_value t = 1;
    k_value epsilon = MathImpl.__epsilon__();

    learningRate = args.size() > 4 ? args.at(4) : learningRate;
    beta1 = args.size() > 5 ? args.at(5) : beta1;
    beta2 = args.size() > 6 ? args.at(6) : beta2;
    t = args.size() > 7 ? args.at(7) : t;
    epsilon = args.size() > 8 ? args.at(8) : epsilon;

    MLOptimizerBuiltins.__adam__(term, weights, gradients, m, v, learningRate,
                                 beta1, beta2, t, epsilon);
    return weights;
  }

  static k_value executeOptimAdamax(const Token& term,
                                    const std::vector<k_value>& args) {
    if (args.size() < 4 || args.size() > 8) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimAdamax);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto m = args.at(2);
    auto v = args.at(3);
    k_value learningRate = 0.02;
    k_value beta1 = 0.9;
    k_value beta2 = 0.999;
    k_value epsilon = MathImpl.__epsilon__();

    learningRate = args.size() > 4 ? args.at(4) : learningRate;
    beta1 = args.size() > 5 ? args.at(5) : beta1;
    beta2 = args.size() > 6 ? args.at(6) : beta2;
    epsilon = args.size() > 7 ? args.at(7) : epsilon;

    MLOptimizerBuiltins.__adamax__(term, weights, gradients, m, v, learningRate,
                                   beta1, beta2, epsilon);
    return weights;
  }

  static k_value executeOptimAdagrad(const Token& term,
                                     const std::vector<k_value>& args) {
    if (args.size() < 3 || args.size() > 5) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimAdagrad);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto v = args.at(2);
    k_value learningRate = 0.01;
    k_value epsilon = MathImpl.__epsilon__();

    learningRate = args.size() > 3 ? args.at(3) : learningRate;
    epsilon = args.size() > 4 ? args.at(4) : epsilon;

    MLOptimizerBuiltins.__adagrad__(term, weights, gradients, v, learningRate,
                                    epsilon);
    return weights;
  }

  static k_value executeOptimAdadelta(const Token& term,
                                      const std::vector<k_value>& args) {
    if (args.size() < 4 || args.size() > 6) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimAdadelta);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto accumGrad = args.at(2);
    auto accumUpdate = args.at(3);
    k_value rho = 0.95;
    k_value epsilon = 1e-6;

    rho = args.size() > 4 ? args.at(4) : rho;
    epsilon = args.size() > 5 ? args.at(5) : epsilon;

    MLOptimizerBuiltins.__adadelta__(term, weights, gradients, accumGrad,
                                     accumUpdate, rho, epsilon);
    return weights;
  }

  static k_value executeOptimRMSProp(const Token& term,
                                     const std::vector<k_value>& args) {
    if (args.size() < 3 || args.size() > 5) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.OptimRMSProp);
    }

    auto weights = args.at(0);
    auto gradients = args.at(1);
    auto v = args.at(2);
    k_value learningRate = 0.001;
    k_value decayRate = 0.9;

    learningRate = args.size() > 3 ? args.at(3) : learningRate;
    decayRate = args.size() > 4 ? args.at(4) : decayRate;

    MLOptimizerBuiltins.__rmsprop__(term, weights, gradients, v, learningRate,
                                    decayRate);
    return weights;
  }

  static k_value executeRegDropout(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegDropout);
    }

    k_value dropoutRate = args.size() > 1 ? args.at(1) : HALF;

    return MLRegularizationBuiltins.__dropout__(term, args.at(0), dropoutRate);
  }

  static k_value executeRegWeightDecay(const Token& term,
                                       const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value decayRate = args.size() > 1 ? args.at(1) : ONE_PERCENT;

    MLRegularizationBuiltins.__weight_decay__(term, args.at(0), decayRate);
    return args.at(0);
  }

  static k_value executeRegL1Lasso(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value lambda = args.size() > 1 ? args.at(1) : ONE_PERCENT;

    return MLRegularizationBuiltins.__l1_regularization__(term, args.at(0),
                                                          lambda);
  }

  static k_value executeRegL2Ridge(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value lambda = args.size() > 1 ? args.at(1) : ONE_PERCENT;

    return MLRegularizationBuiltins.__l2_regularization__(term, args.at(0),
                                                          lambda);
  }

  static k_value executeRegElasticNet(const Token& term,
                                      const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 3) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value lambda1 = args.size() > 1 ? args.at(1) : ONE_PERCENT;
    k_value lambda2 = args.size() > 2 ? args.at(2) : ONE_PERCENT;

    return MLRegularizationBuiltins.__elastic_net__(term, args.at(0), lambda1,
                                                    lambda2);
  }
};

#endif