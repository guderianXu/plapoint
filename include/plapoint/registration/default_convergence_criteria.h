#pragma once

#include <cmath>
#include <limits>
#include <memory>

#include <Eigen/Core>

#include <plapoint/correspondence.h>
#include <plapoint/registration/convergence_criteria.h>

namespace plapoint::registration
{

    template <typename Scalar = float> class DefaultConvergenceCriteria : public ConvergenceCriteria
    {
    public:
        using Ptr = std::shared_ptr<DefaultConvergenceCriteria<Scalar>>;
        using ConstPtr = std::shared_ptr<const DefaultConvergenceCriteria<Scalar>>;
        using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;

        enum ConvergenceState
        {
            CONVERGENCE_CRITERIA_NOT_CONVERGED,
            CONVERGENCE_CRITERIA_ITERATIONS,
            CONVERGENCE_CRITERIA_TRANSFORM,
            CONVERGENCE_CRITERIA_ABS_MSE,
            CONVERGENCE_CRITERIA_REL_MSE,
            CONVERGENCE_CRITERIA_NO_CORRESPONDENCES,
            CONVERGENCE_CRITERIA_FAILURE_AFTER_MAX_ITERATIONS
        };

        DefaultConvergenceCriteria(const int& iterations,
                                   const Matrix4& transformation,
                                   const Correspondences& correspondences)
            : _iterations(iterations), _transformation(transformation), _correspondences(correspondences)
        {
        }

        ~DefaultConvergenceCriteria() override = default;

        void setMaximumIterationsSimilarTransforms(int iterations)
        {
            _maximumSimilarTransforms = iterations;
        }

        int getMaximumIterationsSimilarTransforms() const
        {
            return _maximumSimilarTransforms;
        }

        void setMaximumIterations(int iterations)
        {
            _maximumIterations = iterations;
        }

        int getMaximumIterations() const
        {
            return _maximumIterations;
        }

        void setFailureAfterMaximumIterations(bool failure)
        {
            _failureAfterMaximumIterations = failure;
        }

        bool getFailureAfterMaximumIterations() const
        {
            return _failureAfterMaximumIterations;
        }

        void setRotationThreshold(double threshold)
        {
            _rotationThreshold = threshold;
        }

        double getRotationThreshold() const
        {
            return _rotationThreshold;
        }

        void setTranslationThreshold(double threshold)
        {
            _translationThreshold = threshold;
        }

        double getTranslationThreshold() const
        {
            return _translationThreshold;
        }

        void setRelativeMSE(double threshold)
        {
            _relativeMseThreshold = threshold;
        }

        double getRelativeMSE() const
        {
            return _relativeMseThreshold;
        }

        void setAbsoluteMSE(double threshold)
        {
            _absoluteMseThreshold = threshold;
        }

        double getAbsoluteMSE() const
        {
            return _absoluteMseThreshold;
        }

        bool hasConverged() override
        {
            if (_state != CONVERGENCE_CRITERIA_NOT_CONVERGED)
            {
                _similarTransforms = 0;
                _state = CONVERGENCE_CRITERIA_NOT_CONVERGED;
            }

            bool similar = false;
            if (_iterations >= _maximumIterations)
            {
                _state = _failureAfterMaximumIterations ? CONVERGENCE_CRITERIA_FAILURE_AFTER_MAX_ITERATIONS
                                                        : CONVERGENCE_CRITERIA_ITERATIONS;
                if (!_failureAfterMaximumIterations)
                {
                    return true;
                }
            }

            const double rotation_cosine = 0.5 * static_cast<double>(_transformation(0, 0) + _transformation(1, 1) +
                                                                     _transformation(2, 2) - Scalar(1));
            const double translation_squared = static_cast<double>(_transformation(0, 3)) * _transformation(0, 3) +
                                               static_cast<double>(_transformation(1, 3)) * _transformation(1, 3) +
                                               static_cast<double>(_transformation(2, 3)) * _transformation(2, 3);
            if (rotation_cosine >= _rotationThreshold && translation_squared <= _translationThreshold)
            {
                if (_similarTransforms >= _maximumSimilarTransforms)
                {
                    _state = CONVERGENCE_CRITERIA_TRANSFORM;
                    return true;
                }
                similar = true;
            }

            const double current_mse = calculateMse();
            if (std::isfinite(_previousMse) && std::isfinite(current_mse))
            {
                const double difference = std::abs(current_mse - _previousMse);
                if (difference < _absoluteMseThreshold)
                {
                    if (_similarTransforms >= _maximumSimilarTransforms)
                    {
                        _state = CONVERGENCE_CRITERIA_ABS_MSE;
                        return true;
                    }
                    similar = true;
                }
                if (_previousMse != 0.0 && difference / std::abs(_previousMse) < _relativeMseThreshold)
                {
                    if (_similarTransforms >= _maximumSimilarTransforms)
                    {
                        _state = CONVERGENCE_CRITERIA_REL_MSE;
                        return true;
                    }
                    similar = true;
                }
            }

            _similarTransforms = similar ? _similarTransforms + 1 : 0;
            _previousMse = current_mse;
            return false;
        }

        ConvergenceState getConvergenceState()
        {
            return _state;
        }

        void setConvergenceState(ConvergenceState state)
        {
            _state = state;
        }

        void reset()
        {
            _previousMse = std::numeric_limits<double>::max();
            _similarTransforms = 0;
            _state = CONVERGENCE_CRITERIA_NOT_CONVERGED;
        }

    private:
        double calculateMse() const
        {
            if (_correspondences.empty())
            {
                return std::numeric_limits<double>::infinity();
            }
            double sum = 0.0;
            for (const auto& correspondence : _correspondences)
            {
                sum += correspondence.distance;
            }
            return sum / static_cast<double>(_correspondences.size());
        }

        const int& _iterations;
        const Matrix4& _transformation;
        const Correspondences& _correspondences;
        double _previousMse = std::numeric_limits<double>::max();
        int _maximumIterations = 100;
        bool _failureAfterMaximumIterations = false;
        double _rotationThreshold = 0.99999;
        double _translationThreshold = 3.0e-4 * 3.0e-4;
        double _relativeMseThreshold = 1.0e-5;
        double _absoluteMseThreshold = 1.0e-12;
        int _similarTransforms = 0;
        int _maximumSimilarTransforms = 0;
        ConvergenceState _state = CONVERGENCE_CRITERIA_NOT_CONVERGED;
    };

} // namespace plapoint::registration
