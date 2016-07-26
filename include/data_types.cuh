#ifndef GANJA_DATA_TYPES_CUH
#define GANJA_DATA_TYPES_CUH

#include <type_traits>

template <
    typename index_t,
    index_t bits_key,
    index_t bits_val>
struct KeyValuePair_t {

    static_assert(bits_key+bits_val <= sizeof(index_t)*8,
                  "ERROR: Too many bits for chosen datatype.");
    static_assert(bits_key > 0 && bits_val > 0,
                  "ERROR: All bits must be greater zero.");
    static_assert(std::is_fundamental<index_t>::value,
                  "ERROR: Type index_t must be fundamental type.");
    static_assert(std::is_unsigned<index_t>::value,
                  "ERROR: Type index_t must be unsigned type.");

    typedef KeyValuePair_t<index_t,bits_key,bits_val> data_t;
    typedef std::atomic<data_t> atomic_t;

    static constexpr data_t empty = {~(index_t(0))};
    static constexpr index_t bits_for_key = bits_key;
    static constexpr index_t bits_for_val = bits_val;
    static constexpr index_t mask_key = (1UL << bits_key)-1;
    static constexpr index_t mask_val = (1UL << bits_val)-1;
    static constexpr index_t bytes_for_atomic_storage = sizeof(atomic_t);

    index_t payload;

    void set_pair(
        const index_t& key,
        const index_t& val) {
        payload = key + (val << bits_key);
    }

    void set_key(
        const index_t& key) {
        payload = (payload & ~mask_key) + key;
    }

    void set_val(
        const index_t& val) {
        payload = (payload & ~(mask_val << bits_key)) + (val << bits_key);
    }

    void set_pair_safe(
        const index_t& key,
        const index_t& val) {
        set_pair(key < mask_key ? key : mask_key-1,
                 val < mask_val ? val : mask_val);
    }

    void set_key_safe(
        const index_t& key) {
        set_key(key < mask_key ? key : mask_key-1);
    }

    void set_val_safe(
        const index_t& val) {
        set_val(val < mask_val ? val : mask_val);
    }

    constexpr bool is_lock_free() const {
        return atomic_t::is_lock_free();
    }

    index_t get_key() const {
        return payload & mask_key;
    }

    index_t get_val() const {
        return (payload & (mask_val << bits_val)) >> bits_key;
    }

    inline bool operator==(
        const data_t& other) const {
        return payload == other.payload;
    }

    inline bool operator!=(
        const data_t& other) const {
        return payload != other.payload;
    }
};

template <
    typename index_t,
    index_t bits_key,
    index_t bits_val,
    index_t bits_cnt=sizeof(index_t)*8-bits_key-bits_val>
struct KeyValueCountTriple_t {

    static_assert(bits_key+bits_val+bits_cnt <= sizeof(index_t)*8,
                  "ERROR: Too many bits for chosen datatype.");
    static_assert(bits_key > 0 && bits_val > 0 && bits_cnt > 0,
                  "ERROR: All bits must be greater zero.");
    static_assert(std::is_fundamental<index_t>::value,
                 "ERROR: Type index_t must be fundamental type.");
    static_assert(std::is_unsigned<index_t>::value,
                 "ERROR: Type index_t must be unsigned type.");

    typedef KeyValueCountTriple_t<index_t,bits_key,bits_val,bits_cnt> data_t;
    typedef std::atomic<data_t> atomic_t;

    static constexpr data_t empty = {~(index_t(0))};
    static constexpr index_t bits_for_key = bits_key;
    static constexpr index_t bits_for_val = bits_val;
    static constexpr index_t bits_for_cnt = bits_cnt;
    static constexpr index_t mask_key = (1UL << bits_key)-1;
    static constexpr index_t mask_val = (1UL << bits_val)-1;
    static constexpr index_t mask_cnt = (1UL << bits_cnt)-1;
    static constexpr index_t bytes_for_atomic_storage = sizeof(atomic_t);

    index_t payload;

    void set_triple(
        const index_t& key,
        const index_t& val,
        const index_t& cnt) {
        payload = key + (val << bits_key) + (cnt << (bits_key + bits_val));
    }

    void set_key(
        const index_t& key) {
        payload = (payload & ~mask_key) + key;
    }

    void set_val(
        const index_t& val) {
        payload = (payload & ~(mask_val << bits_key)) + (val << bits_key);
    }

    void set_cnt(
        const index_t& cnt) {
        const index_t offset = bits_key + bits_val;
        payload = (payload & ~(mask_cnt << offset)) + (cnt << offset);
    }

    void set_triple_safe(
        const index_t& key,
        const index_t& val,
        const index_t& cnt) {
        set_triple(key < mask_key ? key : mask_key-1,
                   val < mask_val ? val : mask_val,
                   cnt < mask_cnt ? cnt : mask_cnt);
    }

    void set_key_safe(
        const index_t& key) {
        set_key(key < mask_key ? key : mask_key-1);
    }

    void set_val_safe(
        const index_t& val) {
        set_val(val < mask_val ? val : mask_val);
    }

    void set_cnt_safe(
        const index_t& cnt) {
        set_cnt(cnt < mask_cnt ? cnt : mask_cnt);
    }

    constexpr bool is_lock_free() const {
        return atomic_t::is_lock_free();
    }

    index_t get_key() const {
        return payload & mask_key;
    }

    index_t get_val() const {
        return (payload & (mask_val << bits_val)) >> bits_key;
    }

    index_t get_cnt() const {
        const index_t offset = bits_key + bits_val;
        return (payload & (mask_cnt << offset)) >> offset;
    }

    inline bool operator==(
        const data_t& other) const {
        return payload == other.payload;
    }

    inline bool operator!=(
        const data_t& other) const {
        return payload != other.payload;
    }
};

#endif
