#pragma once

#include "lve_device.hpp"
#include "lve_window.hpp"

namespace lve {

	// Headless MCTS demo. LveDevice needs a window to pick a surface-capable device,
	// but nothing is ever rendered or presented.
	class MctsApp {
	public:
		static constexpr int WIDTH = 320;
		static constexpr int HEIGHT = 240;

		MctsApp() = default;
		void run();

	private:
		LveWindow lveWindow{ WIDTH, HEIGHT, "MCTS (headless)" };
		LveDevice lveDevice{ lveWindow };
	};

}  // namespace lve