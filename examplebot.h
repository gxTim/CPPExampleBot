#pragma once

#include "rlbot/bot.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>

class ExampleBot : public rlbot::Bot {
public:
    ExampleBot(int _index, int _team, std::string _name);
    ~ExampleBot();
    rlbot::Controller GetOutput(rlbot::GameTickPacket gametickpacket) override;

private:
    Ort::Env ort_env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;

    static constexpr float SIDE_WALL_X = 4096.0f;
    static constexpr float BACK_NET_Y = 5120.0f;
    static constexpr float CEILING_Z = 2044.0f;
    static constexpr float CAR_MAX_SPEED = 2300.0f;
    static constexpr float CAR_MAX_ANG_VEL = 5.5f;
    static constexpr float PI_F = 3.14159265f;

    float pos_coef[3] = { 1.0f / SIDE_WALL_X, 1.0f / BACK_NET_Y, 1.0f / CEILING_Z };
    float lin_vel_coef = 1.0f / CAR_MAX_SPEED;
    float ang_vel_coef = 1.0f / CAR_MAX_ANG_VEL;

    std::array<float, 8> prev_action = { 0 };

    void EulerToForwardUp(float pitch, float yaw, float roll,
        float* forward, float* up);
    std::array<float, 8> RunInference(const std::array<float, 89>& obs);
};