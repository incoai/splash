#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::metal {

// A vector that keeps up to Capacity elements in its own storage and moves to
// the heap only past that. Dispatch bindings use it so a dispatch with the
// production binding count costs no allocation. Iterators, pointers and
// references are invalidated by growth, as with std::vector, and also by
// moving an inline vector, since its elements live inside the object.
template <class T, size_t Capacity>
class InlineVector final {
  static_assert(Capacity > 0 && Capacity <= UINT32_MAX,
                "inline capacity must be positive and fit 32 bits");
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "inline elements must move without throwing");

public:
  using value_type = T;
  using size_type = size_t;
  using iterator = T *;
  using const_iterator = const T *;
  using reference = T &;
  using const_reference = const T &;

  InlineVector() noexcept = default;

  InlineVector(std::initializer_list<T> values) { append(values.begin(), values.end()); }

  InlineVector(const InlineVector &other) { append(other.begin(), other.end()); }

  InlineVector(InlineVector &&other) noexcept { take(std::move(other)); }

  InlineVector &operator=(const InlineVector &other) {
    if (this != &other) {
      clear();
      append(other.begin(), other.end());
    }
    return *this;
  }

  InlineVector &operator=(InlineVector &&other) noexcept {
    if (this != &other) {
      release();
      take(std::move(other));
    }
    return *this;
  }

  InlineVector &operator=(std::initializer_list<T> values) {
    clear();
    append(values.begin(), values.end());
    return *this;
  }

  ~InlineVector() { release(); }

  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
  // True while the elements live in this object's own storage.
  [[nodiscard]] bool isInline() const noexcept { return data_ == inlineData(); }

  [[nodiscard]] T *data() noexcept { return data_; }
  [[nodiscard]] const T *data() const noexcept { return data_; }
  [[nodiscard]] iterator begin() noexcept { return data_; }
  [[nodiscard]] iterator end() noexcept { return data_ + size_; }
  [[nodiscard]] const_iterator begin() const noexcept { return data_; }
  [[nodiscard]] const_iterator end() const noexcept { return data_ + size_; }

  [[nodiscard]] T &operator[](size_t index) noexcept { return data_[index]; }
  [[nodiscard]] const T &operator[](size_t index) const noexcept { return data_[index]; }
  [[nodiscard]] T &front() noexcept { return data_[0]; }
  [[nodiscard]] const T &front() const noexcept { return data_[0]; }
  [[nodiscard]] T &back() noexcept { return data_[size_ - 1]; }
  [[nodiscard]] const T &back() const noexcept { return data_[size_ - 1]; }

  void reserve(size_t count) {
    if (count > capacity_)
      grow(count);
  }

  template <class... Arguments>
  T &emplace_back(Arguments &&...arguments) {
    if (size_ == capacity_) {
      // The arguments may refer to an element that growth is about to move.
      T value(std::forward<Arguments>(arguments)...);
      grow(size_t{capacity_} * 2);
      return place(std::move(value));
    }
    return place(std::forward<Arguments>(arguments)...);
  }

  void push_back(const T &value) { emplace_back(value); }
  void push_back(T &&value) { emplace_back(std::move(value)); }

  // Inserts [first, last) before position; appending is the common case. The
  // range must not refer into this vector.
  template <class Iterator>
  iterator insert(const_iterator position, Iterator first, Iterator last) {
    const size_t offset = static_cast<size_t>(position - data_);
    const size_t before = size_;
    append(first, last);
    std::rotate(data_ + offset, data_ + before, data_ + size_);
    return data_ + offset;
  }

  iterator insert(const_iterator position, std::initializer_list<T> values) {
    return insert(position, values.begin(), values.end());
  }

  void clear() noexcept {
    std::destroy(data_, data_ + size_);
    size_ = 0;
  }

  // Drops the elements past count, or appends value-initialized ones up to it.
  void resize(size_t count) {
    if (count <= size_) {
      std::destroy(data_ + count, data_ + size_);
      size_ = static_cast<decltype(size_)>(count);
      return;
    }
    reserve(count);
    while (size_ < count)
      place();
  }

private:
  [[nodiscard]] T *inlineData() noexcept { return reinterpret_cast<T *>(storage_); }
  [[nodiscard]] const T *inlineData() const noexcept {
    return reinterpret_cast<const T *>(storage_);
  }

  template <class... Arguments>
  T &place(Arguments &&...arguments) {
    T *element = std::construct_at(data_ + size_, std::forward<Arguments>(arguments)...);
    ++size_;
    return *element;
  }

  template <class Iterator>
  void append(Iterator first, Iterator last) {
    if constexpr (std::is_base_of_v<std::forward_iterator_tag,
                                    typename std::iterator_traits<Iterator>::iterator_category>)
      reserve(size_ + static_cast<size_t>(std::distance(first, last)));
    for (; first != last; ++first)
      emplace_back(*first);
  }

  // Moves the elements into heap storage of at least count elements.
  void grow(size_t count) {
    if (count > UINT32_MAX)
      throw std::length_error("inline vector capacity exceeds 32 bits");
    std::allocator<T> allocator;
    T *heap = allocator.allocate(count);
    std::uninitialized_move(data_, data_ + size_, heap);
    std::destroy(data_, data_ + size_);
    if (!isInline())
      allocator.deallocate(data_, capacity_);
    data_ = heap;
    capacity_ = static_cast<uint32_t>(count);
  }

  // Leaves other empty and inline; steals heap storage, moves inline elements.
  void take(InlineVector &&other) noexcept {
    if (!other.isInline()) {
      data_ = std::exchange(other.data_, other.inlineData());
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, static_cast<uint32_t>(Capacity));
      return;
    }
    std::uninitialized_move(other.begin(), other.end(), data_);
    size_ = other.size_;
    other.clear();
  }

  // Destroys the elements and returns to empty inline storage.
  void release() noexcept {
    clear();
    if (!isInline())
      std::allocator<T>{}.deallocate(data_, capacity_);
    data_ = inlineData();
    capacity_ = static_cast<uint32_t>(Capacity);
  }

  T *data_ = inlineData();
  uint32_t size_ = 0;
  uint32_t capacity_ = static_cast<uint32_t>(Capacity);
  alignas(T) std::byte storage_[sizeof(T) * Capacity];
};

} // namespace splash::metal
