#pragma once
namespace splash::flash {
// Separate protocol-v5 host; does not alter the existing Qwen-27B worker.
int runFlashWorker(int argc, char **argv);
}
