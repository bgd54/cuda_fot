#ifndef TUPLE_UTILS_HPP_ALSKPXUZ
#define TUPLE_UTILS_HPP_ALSKPXUZ

#include "data_t.hpp"
#include <tuple>
#include <utility>

namespace tuple_utils_details {
namespace redi {
// Copied shamelessly from:
// https://gitorious.org/redistd/redistd/?p=redistd:redistd.git;a=blob;f=include/redi/index_tuple.h;h=3320cdc1bc1d0ead64afe58940051421fb192334;hb=HEAD
/*
 * Boost Software License - Version 1.0 - August 17th, 2003
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Copyright Jonathan Wakely 2012
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/// A type that represents a parameter pack of zero or more integers.
template <unsigned... Indices> struct index_tuple {
  /// Generate an index_tuple with an additional element.
  template <unsigned N> using append = index_tuple<Indices..., N>;
};

/// Unary metafunction that generates an index_tuple containing [0, Size)
template <unsigned Size> struct make_index_tuple {
  typedef typename make_index_tuple<Size - 1>::type::template append<Size - 1>
      type;
};

// Terminal case of the recursive metafunction.
template <> struct make_index_tuple<0u> { typedef index_tuple<> type; };

template <typename... Types>
using to_index_tuple = typename make_index_tuple<sizeof...(Types)>::type;

} // namespace redi

template <class IndexTuple> struct index_tuple_to_tuple;

template <unsigned... Indices>
struct index_tuple_to_tuple<redi::index_tuple<Indices...>> {
  using type = std::tuple<std::integral_constant<unsigned, Indices>...>;
};

template <unsigned I, class... Params> struct _generate_data_set_t;

template <unsigned _DataDim, class _DataType, bool Written> struct Param {};

template <unsigned I, unsigned DataDim, class DataType, bool Written,
          class... Params>
struct _generate_data_set_t<I, Param<DataDim, DataType, Written>, Params...> {
  template <class T, class TT> struct prepend;
  template <class T, class... TupleTypes>
  struct prepend<T, std::tuple<TupleTypes...>> {
    using type = std::tuple<T, TupleTypes...>;
  };

  template <class T, bool _Written, class TT> struct prepend_if {
    using type = TT;
  };

  template <class T, class TT>
  struct prepend_if<T, true, TT> : public prepend<T, TT> {};

  using _tail_decoded = _generate_data_set_t<I + 1, Params...>;
  using data_set_type =
      typename prepend<data_t<DataType, DataDim>,
                       typename _tail_decoded::data_set_type>::type;
  using written_data_set_type =
      typename prepend_if<data_t<DataType, DataDim>, Written,
                          typename _tail_decoded::data_set_type>::type;
  using written_data_set_mapping = typename prepend_if<
      std::integral_constant<unsigned, I>, Written,
      typename _tail_decoded::data_set_type>::written_data_set_mapping;
};

template <unsigned I> struct _generate_data_set_t<I> {
  using data_set_type = std::tuple<>;
  using written_data_set_type = std::tuple<>;
  using written_data_set_mapping = std::tuple<>;
};

template <class... Params> struct DSetParams {};

template <class DSetParams> struct generate_data_set_t;

template <class... Params> struct generate_data_set_t<DSetParams<Params...>> {
  using data_set_type =
      typename _generate_data_set_t<0, Params...>::data_set_type;
  using written_data_set_type =
      typename _generate_data_set_t<0, Params...>::written_data_set_type;
  using written_data_set_mapping =
      typename _generate_data_set_t<0, Params...>::written_data_set_mapping;
};

template <unsigned... Dims> struct _generate_mappings_t {
  using type = std::tuple<data_t<MY_SIZE, Dims>...>;
};

template <class... T, unsigned... I>
std::tuple<T...> initTupleFromArray(const std::array<MY_SIZE, sizeof...(I)> &a,
                                    redi::index_tuple<I...>) {
  return std::tuple<T...>(a[I]...);
}

template <class Tuple> struct InitTupleFromArrayHelper;

template <class... T> struct InitTupleFromArrayHelper<std::tuple<T...>> {
  static std::tuple<T...> call(const std::array<MY_SIZE, sizeof...(T)> &a) {
    return initTupleFromArray<T...>(a, redi::to_index_tuple<T...>{});
  }
};

template <class Func, unsigned I, class... Types> struct for_each;

template <class Func, class... Types>
struct for_each<Func, sizeof...(Types), Types...> {
  static void call(std::tuple<Types...> &, Func) {}
};

template <class Func, unsigned I = 0, class... Types> struct for_each {
  static void call(std::tuple<Types...> &_tuple, Func f) {
    f.operator()<I>(std::get<I>(_tuple));
    for_each<Func, I + 1, Types...>::call(_tuple, f);
  }
};

struct IdentityMapping {
  MY_SIZE operator[](MY_SIZE a) const { return a; }
};

using std::get;

template <unsigned> IdentityMapping get(IdentityMapping a) { return a; }

template <unsigned N, unsigned I, class J_, class LoopBody, class Tuple,
          class Mappings, class WrittenMapping>
struct _call_with_pointers_t {
  static constexpr unsigned J = J_::value;

  template <unsigned MappingDim, unsigned PointOffset> struct helper {
    template <class Mapping, class new_ptr_t, class... pointers_t>
    static void call(Tuple &tuple, const Mappings &mappings, unsigned cell_ind,
                     new_ptr_t base_ptr, const Mapping &mapping,
                     pointers_t... p) {
      new_ptr_t new_ptr =
          base_ptr + mapping[MappingDim * cell_ind + PointOffset];
      helper<MappingDim, PointOffset + 1>::template call(
          tuple, mappings, base_ptr, mapping, p..., new_ptr);
    }
  };

  template <unsigned MappingDim> struct helper<MappingDim, MappingDim> {
    template <class Mapping, class new_ptr_t, class... pointers_t>
    static void call(Tuple &tuple, const Mappings &mappings, unsigned cell_ind,
                     new_ptr_t, const Mapping &, pointers_t... p) {
      _call_with_pointers_t<N, I, std::integral_constant<unsigned, J + 1>,
                            LoopBody, Tuple, Mappings,
                            WrittenMapping>::template call(tuple, mappings,
                                                           cell_ind, p...);
    }
  };

  template <class... pointers_t>
  static void call(Tuple &tuple, const Mappings &mappings, unsigned cell_ind,
                   pointers_t... p) {
    auto base_ptr = std::get<J>(std::get<I>(tuple)).begin();
    using cur_written_mapping =
        typename std::tuple_element<I, WrittenMapping>::type;
    using mapping_ind =
        typename std::tuple_element<J, cur_written_mapping>::type;
    auto mapping = get<mapping_ind::value>(std::get<I>(mappings));
    helper<mapping.dim, 0>::call(tuple, mappings, cell_ind, base_ptr, mapping,
                                 p...);
    //_call_with_pointers_t<N, I, std::integral_constant<unsigned, J + 1>,
    //                      LoopBody, Tuple>::template call(tuple, p..., new_p);
  }
};

template <unsigned N, class J, class LoopBody, class Tuple, class Mappings,
          class WrittenMapping>
struct _call_with_pointers_t<N, N, J, LoopBody, Tuple, Mappings,
                             WrittenMapping> {
  template <class... pointers_t>
  static void call(const Tuple &, const Mappings &, unsigned, pointers_t... p) {
    LoopBody::call(p...);
  }
};

template <unsigned N, unsigned I, class LoopBody, class Tuple, class Mappings,
          class WrittenMapping>
struct _call_with_pointers_t<
    N, I, std::integral_constant<
              unsigned,
              std::tuple_size<typename std::remove_reference<
                  typename std::tuple_element<I, Tuple>::type>::type>::value>,
    LoopBody, Tuple, Mappings, WrittenMapping> {
  template <class... pointers_t>
  static void call(Tuple &tuple, const Mappings &mappings, unsigned cell_ind,
                   pointers_t... p) {
    _call_with_pointers_t<N, I + 1, std::integral_constant<unsigned, 0>,
                          LoopBody, Tuple, Mappings,
                          WrittenMapping>::call(tuple, mappings, cell_ind,
                                                p...);
  }
};

template <class LoopBody, class Tuple, class Mappings, class WrittenMapping>
struct call_with_pointers_t;

template <class LoopBody, class Mappings, class WrittenMapping,
          class... TupleParams>
struct call_with_pointers_t<LoopBody, std::tuple<TupleParams &...>, Mappings,
                            WrittenMapping>
    : public _call_with_pointers_t<
          sizeof...(TupleParams)-1, 0, std::integral_constant<unsigned, 0>,
          LoopBody, std::tuple<TupleParams &...>, Mappings, WrittenMapping> {};

} // namespace tuple_utils_details

/**
 * These two types are used as groupings in template parameter packs.
 */
template <unsigned DataDim, class DataType, bool Written = false>
using Param = tuple_utils_details::Param<DataDim, DataType, Written>;

template <class... Params>
using DSetParams = tuple_utils_details::DSetParams<Params...>;

/**
 * Generate the type of the data_sets: std::tuple of data_t-s of the given sizes
 * and data types.
 */
template <class DSetParams>
using generate_data_set_t =
    tuple_utils_details::generate_data_set_t<DSetParams>;

/**
 * Generate the type of the mappings: std::tuple of data_t-s of the given sizes.
 */
template <unsigned... MeshDims>
using generate_mappings_t =
    typename tuple_utils_details::_generate_mappings_t<MeshDims...>::type;

/**
 * Initialize a tuple from the given array. All the tuple elements should have
 * a ctor with one argument of type MY_SIZE.
 */
// Inspired by:
// https://stackoverflow.com/questions/14089993/how-to-create-initialize-a-stdtuple-from-an-array-when-the-constructors-have
template <class Tuple, class Array> Tuple initTupleFromArray(const Array &a) {
  return tuple_utils_details::InitTupleFromArrayHelper<Tuple>::call(a);
}

template <class Tuple> Tuple initMapping(MY_SIZE v) {
  std::array<MY_SIZE, std::tuple_size<Tuple>::value> a;
  a.fill(v);
  initTupleFromArray<Tuple>(a);
}

/**
 * for_each for mappings/datasets
 *
 * The Func argument is a functor type that has a templated operator() that
 * accepts a data_t and a std::size_t (the index).
 */
// Algorithm inspired by:
// https://stackoverflow.com/questions/1198260/iterate-over-tuple#6894436
template <class Func, class... Types>
void for_each(std::tuple<Types...> &_tuple, Func f) {
  tuple_utils_details::for_each<Func, 0, Types...>::call(_tuple, f);
}

/**
 * Calls LoopBody::call with the pointers extracted from the data_t-s in
 * TupleParams.
 */
template <class LoopBody, class WrittenMapping, class Mappings,
          class DirectDataSet, class IndirectDataSet>
void call_with_pointers(const Mappings &mappings, unsigned cell_ind,
                        const DirectDataSet &cell_weights,
                        const IndirectDataSet &point_weights,
                        IndirectDataSet &point_weights_out) {
  struct _A {
    constexpr int *begin() const { return nullptr; }
  };
  std::tuple<_A> dummy{};
  std::tuple<DirectDataSet &, IndirectDataSet &, IndirectDataSet &,
             std::tuple<_A> &>
      t(cell_weights, point_weights, point_weights_out, dummy);
  std::tuple<tuple_utils_details::IdentityMapping, const Mappings &,
             const Mappings &>
      _mappings(tuple_utils_details::IdentityMapping{}, mappings, mappings);
  using non_written_mapping_indirect =
      typename tuple_utils_details::index_tuple_to_tuple<
          tuple_utils_details::redi::make_index_tuple<
              std::tuple_size<IndirectDataSet>::value>>::type;
  using non_written_mapping_direct =
      typename tuple_utils_details::index_tuple_to_tuple<
          tuple_utils_details::redi::make_index_tuple<
              std::tuple_size<DirectDataSet>::value>>::type;
  using _WrittenMapping =
      std::tuple<non_written_mapping_direct, non_written_mapping_indirect,
                 WrittenMapping>;
  tuple_utils_details::call_with_pointers_t<LoopBody, decltype(t),
                                            decltype(_mappings),
                                            _WrittenMapping>::call(t, _mappings,
                                                                   cell_ind);
}

#endif /* end of include guard: TUPLE_UTILS_HPP_ALSKPXUZ */
// vim:set et sw=2 ts=2 fdm=marker:
