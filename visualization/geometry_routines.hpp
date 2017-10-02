#ifndef GEOMETRY_ROUTINES_HPP_EJHD6TA2
#define GEOMETRY_ROUTINES_HPP_EJHD6TA2

#include <array>
#include <cmath>

namespace geom {

// enum TurnDirection : int {
//  LEFT = -1,
//  STRAIGHT = 0,
//  RIGHT = 1
//};

namespace two_d {

struct Vector {
  float x, y;
};

struct LineSegment {
  Vector start, end;
};

inline float turns(Vector a, Vector b, Vector c) {
  Vector s1 = {b.x - a.x, b.y - a.y};
  Vector s2 = {c.x - b.x, c.y - b.y};
  float r = s1.y * s2.x - s1.x * s2.y;
  return r;
  // return static_cast<TurnDirection>((0 < r) - (r < 0));
}

inline bool crosses(LineSegment a, LineSegment b) {
  return turns(a.start, a.end, b.start) * turns(a.start, a.end, b.end) < 0 &&
         turns(b.start, b.end, a.start) * turns(b.start, b.end, a.end) < 0;
}

inline std::array<MY_SIZE, 4> reorderQuad(std::array<float, 12> points) {
  LineSegment a{{points[0], points[1]}, {points[3], points[4]}};
  LineSegment b{{points[3], points[4]}, {points[6], points[7]}};
  LineSegment c{{points[6], points[7]}, {points[9], points[10]}};
  LineSegment d{{points[9], points[10]}, {points[0], points[1]}};
  if (crosses(a, c)) {
    return {{0, 2, 1, 3}};
  } else if (crosses(b, d)) {
    return {{0, 1, 3, 2}};
  }
  return {{0, 1, 2, 3}};
}
}
}

#endif /* end of include guard: GEOMETRY_ROUTINES_HPP_EJHD6TA2 */
