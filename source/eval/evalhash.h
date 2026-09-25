#ifndef EVALHASH_H_INCLUDED
#define EVALHASH_H_INCLUDED

#include "../types.h"
#include "../misc.h"
#include <atomic>
#include <new>

namespace YaneuraOu {

// シンプルなHashTableの実装。Sizeは2のべき乗。
// 評価値のcacheに用いる。
template <typename T>
struct HashTable
{
	// 配列のresize。単位は[MB]
	void resize(ThreadPool& threads, size_t mbSize)
	{
		size_t newClusterCount = mbSize * 1024 * 1024 / sizeof(T);
		newClusterCount = (size_t)1 << MSB64(newClusterCount); // msbだけ取り、2**nであることを保証する

		if (newClusterCount != size)
		{
			release();
			size = newClusterCount;

			// ゼロクリアしておかないと、benchの結果が不安定になる。
			// 気持ち悪いのでゼロクリアしておく。
			entries_ = (T*)aligned_large_pages_alloc(size * sizeof(T));
			clear(threads);
		}
	}

	void release()
	{
		if (entries_)
		{
			aligned_large_pages_free(entries_);
			entries_ = nullptr;
		}
	}

	~HashTable() { release(); }

	T* operator[] (const Key k) { return entries_ + (static_cast<size_t>(k) & (size - 1)); }
	void clear(ThreadPool& threads) { Tools::memclear(threads, "eHash", entries_, size * sizeof(T)); }
	size_t entry_count() const { return size; }
	size_t byte_size() const { return size * sizeof(T); }

private:

	size_t size = 0;
	T* entries_ = nullptr;
};

#if defined(EVAL_HASH_ATOMIC64)
// A direct-mapped table whose complete shared entry is one atomic word.
// It deliberately does not publish or protect any other object.
struct Atomic64HashTable {
    void resize(ThreadPool&, size_t mbSize) {
        size_t requestedBytes = mbSize * 1024 * 1024;
#if defined(EVAL_HASH_SIZE_DIVISOR)
        static_assert(EVAL_HASH_SIZE_DIVISOR >= 1);
        requestedBytes /= EVAL_HASH_SIZE_DIVISOR;
#endif
#if defined(EVAL_HASH_ATOMIC64_QUARTER_BYTES)
        requestedBytes /= 4; // diagnostic size sweep: 1 MiB option => 256 KiB.
#elif defined(EVAL_HASH_ATOMIC64_EIGHTH_BYTES)
        requestedBytes /= 8; // diagnostic size sweep: 1 MiB option => 128 KiB.
#endif
        size_t newCount = requestedBytes / sizeof(std::atomic<std::uint64_t>);
        newCount = size_t(1) << MSB64(newCount);
        if (newCount == size) return;
        release();
        size = newCount;
        entries_ = static_cast<std::atomic<std::uint64_t>*>(
          aligned_large_pages_alloc(size * sizeof(std::atomic<std::uint64_t>)));
        for (size_t i=0;i<size;++i)
            ::new (static_cast<void*>(entries_+i)) std::atomic<std::uint64_t>(0);
    }
    void clear(ThreadPool&) {
        for (size_t i=0;i<size;++i) entries_[i].store(0, std::memory_order_relaxed);
    }
    void release() {
        if (!entries_) return;
        aligned_large_pages_free(entries_); entries_=nullptr; size=0;
    }
    ~Atomic64HashTable() { release(); }
    std::atomic<std::uint64_t>& operator[](Key key) {
        return entries_[static_cast<size_t>(key)&(size-1)];
    }
    size_t entry_count() const { return size; }
    size_t byte_size() const { return size*sizeof(std::atomic<std::uint64_t>); }
private:
    size_t size=0;
    std::atomic<std::uint64_t>* entries_=nullptr;
};
#endif

} // namespace YaneuraOu

#endif // EVALHASH_H_INCLUDED
