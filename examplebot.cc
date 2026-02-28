#include "examplebot.h"
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>

ExampleBot::ExampleBot(int _index, int _team, std::string _name)
    : Bot(_index, _team, _name),
    ort_env(ORT_LOGGING_LEVEL_WARNING, "RLBot") {

    std::cout << "=== ML Bot Konstruktor gestartet ===" << std::endl;
    std::cout << "Index: " << _index << " Team: " << _team << " Name: " << _name << std::endl;

    try {
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        session = std::make_unique<Ort::Session>(ort_env, L"model.onnx", sessionOptions);
#else
        session = std::make_unique<Ort::Session>(ort_env, "model.onnx", sessionOptions);
#endif
        std::cout << "=== ONNX Modell geladen! ===" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Fehler: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Fehler: " << e.what() << std::endl;
    }
}

ExampleBot::~ExampleBot() {}

void ExampleBot::EulerToForwardUp(float pitch, float yaw, float roll,
    float* forward, float* up) {
    float cp = std::cos(pitch);
    float sp = std::sin(pitch);
    float cy = std::cos(yaw);
    float sy = std::sin(yaw);
    float cr = std::cos(roll);
    float sr = std::sin(roll);

    forward[0] = cp * cy;
    forward[1] = cp * sy;
    forward[2] = sp;

    up[0] = -cr * cy * sp - sr * sy;
    up[1] = -cr * sy * sp + sr * cy;
    up[2] = cp * cr;
}

std::array<float, 8> ExampleBot::RunInference(
    const std::array<float, 89>& obs) {

    std::array<int64_t, 2> inputShape = { 1, 89 };
    auto memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    auto inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float*>(obs.data()), 89,
        inputShape.data(), 2);

    const char* inputNames[] = { "observation" };
    const char* outputNames[] = { "action" };

    auto output = session->Run(Ort::RunOptions{ nullptr },
        inputNames, &inputTensor, 1, outputNames, 1);

    float* outData = output[0].GetTensorMutableData<float>();
    std::array<float, 8> actions;
    std::copy(outData, outData + 8, actions.begin());
    return actions;
}

rlbot::Controller ExampleBot::GetOutput(
    rlbot::GameTickPacket gametickpacket) {

    try {
        if (!session) {
            std::cerr << "ONNX Session nicht geladen, leerer Controller wird zurueckgegeben" << std::endl;
            rlbot::Controller controller{ 0 };
            return controller;
        }

        auto* ball = gametickpacket->ball();
        if (!ball || !ball->physics()) {
            rlbot::Controller controller{ 0 };
            return controller;
        }
        auto* ballPhys = ball->physics();
        auto* ballLoc = ballPhys->location();
        auto* ballVel = ballPhys->velocity();
        auto* ballAngVel = ballPhys->angularVelocity();

        if (!gametickpacket->players() ||
            static_cast<int>(gametickpacket->players()->size()) <= index) {
            rlbot::Controller controller{ 0 };
            return controller;
        }

        auto* player = gametickpacket->players()->Get(index);
        if (!player || !player->physics()) {
            rlbot::Controller controller{ 0 };
            return controller;
        }
        auto* playerPhys = player->physics();
        auto* playerLoc = playerPhys->location();
        auto* playerVel = playerPhys->velocity();
        auto* playerAngVel = playerPhys->angularVelocity();
        auto* playerRot = playerPhys->rotation();

        int enemyIdx = -1;
        for (uint32_t i = 0; i < gametickpacket->players()->size(); i++) {
            if ((int)i != index) {
                enemyIdx = i;
                break;
            }
        }

        float inv = (team == 1) ? -1.0f : 1.0f;

        std::array<float, 89> obs = {};
        int idx = 0;

        // Ball Position [0-2]
        obs[idx++] = ballLoc->x() * inv * pos_coef[0];
        obs[idx++] = ballLoc->y() * inv * pos_coef[1];
        obs[idx++] = ballLoc->z() * pos_coef[2];

        // Ball Linear Velocity [3-5]
        obs[idx++] = ballVel->x() * inv * lin_vel_coef;
        obs[idx++] = ballVel->y() * inv * lin_vel_coef;
        obs[idx++] = ballVel->z() * lin_vel_coef;

        // Ball Angular Velocity [6-8] — ang_vel_coef = 1/π
        obs[idx++] = ballAngVel->x() * inv * ang_vel_coef;
        obs[idx++] = ballAngVel->y() * inv * ang_vel_coef;
        obs[idx++] = ballAngVel->z() * ang_vel_coef;

        // Previous Action [9-16]
        for (int i = 0; i < 8; i++) {
            obs[idx++] = prev_action[i];
        }

        // Boost Pads [17-50] (34 pads)
        for (int i = 0; i < 34; i++) {
            obs[idx++] = 1.0f;
        }

        // Player
        float pForward[3], pUp[3];
        EulerToForwardUp(playerRot->pitch(), playerRot->yaw(), playerRot->roll(),
            pForward, pUp);

        obs[idx++] = playerLoc->x() * inv * pos_coef[0];
        obs[idx++] = playerLoc->y() * inv * pos_coef[1];
        obs[idx++] = playerLoc->z() * pos_coef[2];

        obs[idx++] = pForward[0] * inv;
        obs[idx++] = pForward[1] * inv;
        obs[idx++] = pForward[2];

        obs[idx++] = pUp[0] * inv;
        obs[idx++] = pUp[1] * inv;
        obs[idx++] = pUp[2];

        obs[idx++] = playerVel->x() * inv * lin_vel_coef;
        obs[idx++] = playerVel->y() * inv * lin_vel_coef;
        obs[idx++] = playerVel->z() * lin_vel_coef;

        obs[idx++] = playerAngVel->x() * inv * ang_vel_coef;
        obs[idx++] = playerAngVel->y() * inv * ang_vel_coef;
        obs[idx++] = playerAngVel->z() * ang_vel_coef;

        obs[idx++] = player->boost() / 100.0f;
        obs[idx++] = 1.0f;  // on_ground
        obs[idx++] = 1.0f;  // has_flip
        obs[idx++] = player->isDemolished() ? 1.0f : 0.0f;

        // Enemy
        if (enemyIdx >= 0) {
            auto* enemy = gametickpacket->players()->Get(enemyIdx);
            auto* ePhys = enemy->physics();
            auto* eLoc = ePhys->location();
            auto* eVel = ePhys->velocity();
            auto* eAngVel = ePhys->angularVelocity();
            auto* eRot = ePhys->rotation();

            float eForward[3], eUp[3];
            EulerToForwardUp(eRot->pitch(), eRot->yaw(), eRot->roll(),
                eForward, eUp);

            obs[idx++] = eLoc->x() * inv * pos_coef[0];
            obs[idx++] = eLoc->y() * inv * pos_coef[1];
            obs[idx++] = eLoc->z() * pos_coef[2];

            obs[idx++] = eForward[0] * inv;
            obs[idx++] = eForward[1] * inv;
            obs[idx++] = eForward[2];

            obs[idx++] = eUp[0] * inv;
            obs[idx++] = eUp[1] * inv;
            obs[idx++] = eUp[2];

            obs[idx++] = eVel->x() * inv * lin_vel_coef;
            obs[idx++] = eVel->y() * inv * lin_vel_coef;
            obs[idx++] = eVel->z() * lin_vel_coef;

            obs[idx++] = eAngVel->x() * inv * ang_vel_coef;
            obs[idx++] = eAngVel->y() * inv * ang_vel_coef;
            obs[idx++] = eAngVel->z() * ang_vel_coef;

            obs[idx++] = enemy->boost() / 100.0f;
            obs[idx++] = 1.0f;
            obs[idx++] = 1.0f;
            obs[idx++] = enemy->isDemolished() ? 1.0f : 0.0f;
        }

        // Inferenz
        auto actions = RunInference(obs);

        // Debug: alle ~1 Sekunde Actions ausgeben
        if (frame_counter++ % 120 == 0) {
            std::cout << "Actions: "
                << actions[0] << " " << actions[1] << " " << actions[2] << " "
                << actions[3] << " " << actions[4] << " " << actions[5] << " "
                << actions[6] << " " << actions[7] << std::endl;
        }

        // Controller
        rlbot::Controller controller{ 0 };
        controller.throttle = std::clamp(actions[0], -1.0f, 1.0f);
        controller.steer    = std::clamp(actions[1], -1.0f, 1.0f);
        controller.yaw      = std::clamp(actions[2], -1.0f, 1.0f);  // WAR pitch!
        controller.pitch    = std::clamp(actions[3], -1.0f, 1.0f);  // WAR yaw!
        controller.roll     = std::clamp(actions[4], -1.0f, 1.0f);
        controller.jump     = actions[5] > 0.0f;
        controller.boost    = actions[6] > 0.0f;
        controller.handbrake = actions[7] > 0.0f;
        prev_action = actions;

        return controller;

    } catch (const std::exception& e) {
        std::cerr << "GetOutput FEHLER: " << e.what() << std::endl;
        rlbot::Controller controller{ 0 };
        return controller;
    }
}