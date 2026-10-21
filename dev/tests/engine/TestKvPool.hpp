#pragma once

#include "engine/KvPool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace splash::test {

// Extents of extentPages pages, none allocated until the pool allocates
// them (its runway first), as PageStorage, of bytesPerPage bytes a Full page
// and indexBytesPerPage an Index page; without index bytes, every page is
// Full. Growth is refused with allocationFailure (EngineBudget, the
// governor's cause, unless a test names another) past the bytes of
// budgetPages Full pages, while growthBlocked, or when growthAllowed(extent)
// says no; a budget-limited pool needs more page ids than its budget, as
// production's has. A release or a copy throws std::logic_error while
// commandInFlight() says a command is in flight, as PageStorage's does.
// Every page holds one value, which a test sets, a copy carries (between
// forms too, as the page's index) and a clear zeroes, so a test can follow
// the content of pages the pool moves.
class TestKvStorage final : public kv::ExtentStorage {
public:
    TestKvStorage(uint32_t pages, uint64_t bytesPerPage, uint32_t extentPages,
                  uint64_t indexBytesPerPage = 0)
        : content(pages), pageCount_(pages), bytesPerPage_(bytesPerPage),
          indexBytesPerPage_(indexBytesPerPage), extentPages_(extentPages) {
        if (!pages || !bytesPerPage || !extentPages || pages % extentPages) {
            throw std::invalid_argument("invalid test KV extent storage");
        }
        forms_.resize(pages / extentPages);
    }

    uint32_t pageCount() const noexcept override { return pageCount_; }
    uint64_t bytesPerPage(kv::PageForm form) const noexcept override {
        return form == kv::PageForm::Full ? bytesPerPage_ : indexBytesPerPage_;
    }
    uint32_t extentPages() const noexcept override { return extentPages_; }
    metal::AllocationResult allocateExtent(uint32_t extent, kv::PageForm form) override {
        ++allocationAttempts;
        if (forms_.at(extent))
            throw std::logic_error("test KV extent is already allocated");
        if (form == kv::PageForm::Index && !indexBytesPerPage_)
            throw std::invalid_argument("test KV pages without an index are all Full");
        if (growthBlocked || (growthAllowed && !growthAllowed(extent)) ||
            allocatedBytes() + extentPages_ * bytesPerPage(form) > budgetPages * bytesPerPage_) {
            return allocationFailure;
        }
        forms_[extent] = form;
        return {};
    }
    void releaseExtent(uint32_t extent) override {
        if (!forms_.at(extent))
            throw std::logic_error("test KV extent is not allocated");
        if (commandInFlight && commandInFlight()) {
            throw std::logic_error(
                "cannot release a KV extent while a command is in flight");
        }
        forms_[extent].reset();
        const auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(releaseTime);
        const double milliseconds = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - start)
                                        .count();
        longestRelease = std::max(longestRelease, milliseconds);
        totalRelease += milliseconds;
        ++releasedExtents;
        if (released) released();
    }
    void copyPages(std::span<const kv::PageCopy> pages) override {
        if (commandInFlight && commandInFlight()) {
            throw std::logic_error(
                "cannot copy KV pages while a command is in flight");
        }
        for (const kv::PageCopy &copy : pages) {
            content.at(copy.to) = content.at(copy.from);
            copies.push_back(copy);
        }
    }
    void clearPages(std::span<const uint32_t> pages) override {
        for (uint32_t page : pages) {
            if (!forms_.at(page / extentPages_)) {
                throw std::logic_error(
                    "cannot clear a KV page of an extent that is not allocated");
            }
        }
        for (uint32_t page : pages) {
            content.at(page) = 0;
            cleared.push_back(page);
        }
    }

    [[nodiscard]] bool allocated(uint32_t extent) const {
        return forms_.at(extent).has_value();
    }
    [[nodiscard]] uint32_t allocatedPages() const noexcept {
        uint32_t extents = 0;
        for (const auto &form : forms_) extents += form.has_value();
        return extents * extentPages_;
    }
    [[nodiscard]] uint64_t allocatedBytes() const noexcept {
        uint64_t bytes = 0;
        for (const auto &form : forms_)
            if (form) bytes += extentPages_ * bytesPerPage(*form);
        return bytes;
    }

    uint32_t budgetPages = std::numeric_limits<uint32_t>::max();
    bool growthBlocked = false;
    std::function<bool(uint32_t extent)> growthAllowed;
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::EngineBudget;
    std::function<bool()> commandInFlight;
    // After each release: a budget shared with states returns its bytes.
    std::function<void()> released;
    // How long releasing one extent takes.
    std::chrono::milliseconds releaseTime{0};
    uint32_t allocationAttempts = 0;
    uint32_t releasedExtents = 0;
    // What the releases took as measured here, in milliseconds: the longest,
    // and all of them together.
    double longestRelease = 0.0;
    double totalRelease = 0.0;
    std::vector<uint64_t> content;
    std::vector<kv::PageCopy> copies;
    std::vector<uint32_t> cleared;

private:
    uint32_t pageCount_ = 0;
    uint64_t bytesPerPage_ = 0;
    uint64_t indexBytesPerPage_ = 0;
    uint32_t extentPages_ = 0;
    // The form of each allocated extent.
    std::vector<std::optional<kv::PageForm>> forms_;
};

}  // namespace splash::test
