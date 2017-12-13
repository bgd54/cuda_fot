#include "priority_queue.hpp"
#include <catch.hpp>

TEST_CASE("Empty PriorityQueue", "[priority queue]") {
  PriorityQueue<int> pq{};
  auto r = pq.getPriority(4);
  CHECK(r.first == false);
}

TEST_CASE("PriorityQueue: push", "[priority queue]") {
  PriorityQueue<int> pq{};
  pq.push(0, 1);
  SECTION("push one") {
    auto r = pq.getPriority(0);
    CHECK(r.first);
    CHECK(r.second == 1);
  }
  SECTION("push two") {
    pq.push(2, 3);
    auto r = pq.getPriority(0);
    CHECK(r.first);
    CHECK(r.second == 1);
    r = pq.getPriority(2);
    CHECK(r.first);
    CHECK(r.second == 3);
    r = pq.getPriority(1);
    CHECK(r.first == false);
    r = pq.getPriority(3);
    CHECK(r.first == false);
  }
  SECTION("push two w/ same priority") {
    pq.push(2, 1);
    auto r = pq.getPriority(0);
    CHECK(r.first);
    CHECK(r.second == 1);
    r = pq.getPriority(2);
    CHECK(r.first);
    CHECK(r.second == 1);
    r = pq.getPriority(1);
    CHECK(r.first == false);
  }
}

TEST_CASE("PriorityQueue: popMax", "[priority queue]") {
  PriorityQueue<int> pq{};
  pq.push(0, 1);
  SECTION("push one") {
    auto r = pq.popMax();
    CHECK(r.first == 0);
    CHECK(r.second == 1);
    auto rr = pq.getPriority(0);
    CHECK(rr.first == false);
  }
  SECTION("push two") {
    pq.push(2, 3);
    // Pop one
    auto r = pq.popMax();
    CHECK(r.first == 2);
    CHECK(r.second == 3);
    auto rr = pq.getPriority(0);
    CHECK(rr.first);
    CHECK(rr.second == 1);
    rr = pq.getPriority(2);
    CHECK(rr.first == false);
    // And the other
    r = pq.popMax();
    CHECK(r.first == 0);
    CHECK(r.second == 1);
    rr = pq.getPriority(0);
    CHECK(rr.first == false);
    rr = pq.getPriority(2);
    CHECK(rr.first == false);
  }
}

TEST_CASE("PriorityQueue: priority sort", "[priority_queue]") {
  PriorityQueue<int> pq{};
  std::vector<int> v = {1, 9, 8, 0, 7, 2, 6, 4, 3, 11, 12, 13, 10, 5};
  for (unsigned i = 0; i < v.size(); ++i) {
    pq.push(i, v[i]);
  }
  int old = v.size();
  for (unsigned i = 0; i < v.size(); ++i) {
    auto r = pq.popMax();
    INFO("at i == " << i);
    CHECK(r.second == old - 1);
    old = r.second;
  }
}

TEST_CASE("PriorityQueue: modify", "[priority queue]") {
  PriorityQueue<int> pq{};
  pq.push(0, 1);
  pq.push(2, 3);
  SECTION("increase") {
    pq.modify(2, 4);
    auto r = pq.getPriority(2);
    CHECK(r.first);
    CHECK(r.second == 4);
    r = pq.getPriority(0);
    CHECK(r.first);
    CHECK(r.second == 1);
    auto rr = pq.popMax();
    CHECK(rr.first == 2);
    CHECK(rr.second == 4);
  }
  SECTION("decrease") {
    pq.modify(0, -1);
    auto r = pq.getPriority(2);
    CHECK(r.first);
    CHECK(r.second == 3);
    r = pq.getPriority(0);
    CHECK(r.first);
    CHECK(r.second == -1);
    auto rr = pq.popMax();
    CHECK(rr.first == 2);
    CHECK(rr.second == 3);
  }
  SECTION("reverse order") {
    pq.modify(0, 4);
    auto r = pq.getPriority(2);
    CHECK(r.first);
    CHECK(r.second == 3);
    r = pq.getPriority(0);
    CHECK(r.first);
    CHECK(r.second == 4);
    auto rr = pq.popMax();
    CHECK(rr.first == 0);
    CHECK(rr.second == 4);
  }
}
