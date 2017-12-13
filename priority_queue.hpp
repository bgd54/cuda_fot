#ifndef PRIORITY_QUEUE_HPP_VS5TTLHK
#define PRIORITY_QUEUE_HPP_VS5TTLHK

#include <cassert>
#include <map>

template <class Priority> class PriorityQueue {
  std::map<MY_SIZE, typename std::map<Priority, MY_SIZE>::iterator>
      index_to_priority;
  std::multimap<Priority, MY_SIZE> priority_to_index;

public:
  void push(MY_SIZE index, Priority priority);
  std::pair<MY_SIZE, Priority> popMax();
  void modify(MY_SIZE index, Priority new_priority);
  std::pair<bool, Priority> getPriority(MY_SIZE index) const {
    auto it = index_to_priority.find(index);
    if (it == index_to_priority.end()) {
      return std::make_pair(false, Priority{});
    }
    return std::make_pair(true, it->second->first);
  }

  PriorityQueue() : index_to_priority{}, priority_to_index{} {}

  PriorityQueue(const PriorityQueue &other) = delete;
  PriorityQueue &operator=(const PriorityQueue &rhs) = delete;
  PriorityQueue(PriorityQueue &&other) = default;
  PriorityQueue &operator=(PriorityQueue &&rhs) = default;
};

template <class Priority>
void PriorityQueue<Priority>::push(MY_SIZE index, Priority priority) {
  auto it = priority_to_index.insert(std::make_pair(priority, index));
  assert(index_to_priority.find(index) == index_to_priority.end());
  index_to_priority[index] = it;
}

template <class Priority>
std::pair<MY_SIZE, Priority> PriorityQueue<Priority>::popMax() {
  assert(!index_to_priority.empty());
  assert(index_to_priority.size() == priority_to_index.size());
  auto max_reverse_it = priority_to_index.rbegin();
  auto max_it = (++max_reverse_it).base();
  index_to_priority.erase(max_it->second);
  priority_to_index.erase(max_it);
  return std::make_pair(max_it->second, max_it->first);
}

template <class Priority>
void PriorityQueue<Priority>::modify(MY_SIZE index, Priority new_priority) {
  auto it = index_to_priority.find(index);
  assert(it != index_to_priority.end());
  priority_to_index.erase(it->second);
  auto new_it = priority_to_index.insert(std::make_pair(new_priority, index));
  it->second = new_it;
}

#endif /* end of include guard: PRIORITY_QUEUE_HPP_VS5TTLHK */
