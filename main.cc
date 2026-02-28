#include "rlbot/rlbot_generated.h"

#include "examplebot.h"

#include "rlbot/bot.h"
#include "rlbot/botmanager.h"
#include "rlbot/interface.h"
#include "rlbot/platform.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

uint16_t getPortFromFile(std::string filename) {
  std::ifstream file;
  file.open(filename);

  if (!file.is_open()) {
    std::cerr << "Could not open " << filename
              << ", using default port 12345" << std::endl;
    return 12345;
  }

  std::string line;
  std::getline(file, line);
  file.close();

  try {
    return static_cast<uint16_t>(std::stoi(line));
  } catch (const std::exception &e) {
    std::cerr << "Invalid port in " << filename << ": " << e.what()
              << ", using default port 12345" << std::endl;
    return 12345;
  }
}

rlbot::Bot *botFactory(int index, int team, std::string name) {
  return new ExampleBot(index, team, name);
}

int main(int argc, char **argv) {
  // Set the working directory to the directory of this executable so we can use
  // relative paths.
  rlbot::platform::SetWorkingDirectory(
      rlbot::platform::GetExecutableDirectory());

  // Read the port that we use for receiving bot spawn messages.
  uint16_t port = getPortFromFile("port.cfg");

  // Start the bot server.
  rlbot::BotManager botmanager(botFactory);
  botmanager.StartBotServer(port);

  return 0;
}
