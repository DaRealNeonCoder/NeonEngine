#include "mcts_app.hpp"

#include "mcts_gpu.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

namespace lve {

    namespace {

        constexpr int COLS = MctsGpu::COLS;
        constexpr int ROWS = MctsGpu::ROWS;
        using Board = std::array<int, MctsGpu::CELLS>;

        // rows[0] is the TOP row. 'X' = +1 (moves first), 'O' = -1, '.' = empty.
        Board parseBoard(const std::array<std::string, ROWS>& rows) {
            Board b{};
            for (int r = 0; r < ROWS; r++) {
                const std::string& line = rows[ROWS - 1 - r];  // flip so row 0 = bottom
                for (int c = 0; c < COLS; c++) {
                    b[r * COLS + c] = (line[c] == 'X') ? 1 : (line[c] == 'O') ? -1 : 0;
                }
            }
            return b;
        }

        void runTest(
            MctsGpu& mcts,
            const char* name,
            const Board& board,
            int turn,
            int iterations,
            int expected) {
            std::cout << "\n=== " << name << " (" << (turn == 1 ? "X" : "O") << " to move) ===\n";
            for (int r = ROWS - 1; r >= 0; r--) {
                for (int c = 0; c < COLS; c++) {
                    int v = board[r * COLS + c];
                    std::cout << (v == 1 ? 'X' : v == -1 ? 'O' : '.') << ' ';
                }
                std::cout << '\n';
            }

            auto res = mcts.search(board, turn, iterations);

            std::cout << "col  visits    winrate\n";
            for (int c = 0; c < COLS; c++) {
                std::cout << "  " << c << "  " << std::setw(8) << res.visits[c] << "   " << std::fixed
                    << std::setprecision(3) << res.winRate[c] << '\n';
            }
            std::cout << "best column: " << res.bestAction;
            if (expected >= 0) std::cout << (res.bestAction == expected ? "  [OK]" : "  [UNEXPECTED]");
            std::cout << "   (" << iterations << " iterations, " << std::setprecision(1) << res.ms
                << " ms)\n";
        }

    }  // namespace

    void MctsApp::run() {
        MctsGpu::Config cfg;
        cfg.numTrees = 128;
        cfg.maxIterations = 500;
        cfg.shaderPath = "shaders/mcts_connect4.spv";  // adjust to your path
        MctsGpu mcts{ lveDevice, cfg };

        // 1. empty board: center column is the classic best opening
        Board empty{};
        runTest(mcts, "Empty board", empty, +1, 500, 3);

        // 2. X can win immediately in column 3
        Board win = parseBoard({
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "XXX.OOO" });
        runTest(mcts, "X wins in col 3", win, +1, 200, 3);

        // 3. O threatens four in a row at col 3; X must block
        Board block = parseBoard({
            ".......",
            ".......",
            ".......",
            ".......",
            "....X..",
            "OOO.XX." });
        runTest(mcts, "X must block col 3", block, +1, 200, 3);

        std::cout << "\nDone.\n";
    }

}  // namespace lve