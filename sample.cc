__global__ void sync_test(int *ng /* out int*/, int *comm /*in array == 0*/, int *res /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  ng = 0;
  __syncthreads();
  if (threadIdx % 2)
  {
    coalesced_group cg = coalesced_threads();
    unsigned int group_idx;
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
    }
    group_idx = cg.shfl(group_idx, 0);

    if (leader)
    {
      comm[group_idx] = 1;
    }
    cg.sync();
    *res[threadIdx] = comm[group_idx] == 1;
  }
}

__global__ void num_rank_test(int *ng /*out int*/, int *res /*out array*/, int *ptr /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadIdx % 2)
  {
    coalesced_group cg = coalesced_threads();
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      atomicInc(&ng, UINT_MAX);
    }
    res[threadIdx] = cg.num_threads(); // [2,2,3,2,3,2,3]
    ptr[threadIdx] = cg.thread_rank(); // [0,0,0,1,1,1,2] or [0,1,0,0,1,1,2]
  }
}

__global__ void shfl_up_down_test(int *ng /*out int*/, int *res /*out array*/, int *sum /*out array*/, int *input /*in array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int localSum = input[threadIdx];
  ng = 0;
  int x = 0;
  int power = 2;
  __syncthreads();
  if (threadIdx % 2)
  {
    coalesced_group cg = coalesced_threads();
    bool leader = cg.thread_rank() == 0;
    unsigned int group_idx;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
      // calcuate closest power of 2 larger than the cg.num_threads
      x = cg.num_threads();
      x--;
      while (x >>= 1)
      {
        power <<= 1;
      }
    }

    // Version 1 for shfl_up
    for (int i = 0; i < power; i = power / 2)
    {
      atomicAdd(localSum, cg.shfl_up(localSum, i));
    }

    // Version 2 for shfl_down
    for (int i = cg.num_threads(); i > 0; i = power / 2)
    {
      atomicAdd(localSum, cg.shfl_down(localSum, i));
    }
    cg.sync();

    if (leader)
    {
      sum[group_idx] = localSum;
    }
  }
}

__global__ void all_test(int *ng /*out int*/, int *out /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  __syncthreads();

  if (threadIdx % 2)
  {
    unsigned int group_idx;

    coalesced_group cg = coalesced_threads();
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
      *out[group_idx] = cg.all(threadIdx % 2 == 0); // threadIdx % 2 == 1 for true
    }
  }
}

__global__ void any_test(int *ng /*out int*/, int *out /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int val = 2;
  __syncthreads();

  if (threadIdx % 2)
  {
    unsigned int group_idx;

    coalesced_group cg = coalesced_threads();
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
      val = 3; // We can only know for sure the coalesced group will contain at least one thread
    }
    if (leader)
    {
      *out[group_idx] = cg.any(val % 2 == 1); // true
    }
  }
}

__global__ void ballot_test(int *ng /*out int*/, int *out /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int val;
  __syncthreads();

  if (threadIdx % 2)
  {
    unsigned int group_idx;

    coalesced_group cg = coalesced_threads();
    if (cg.thread_rank() == 0)
    {
      val = 1;
    }
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
      *out[group_idx] = cg.ballot(val); // = 1 (We can only know for sure the coalesced group will contain at least one thread)
    }
  }
}

__global__ void match_any_test(int *ng /*out int*/, int *out /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int val = threadIdx;
  int laneId;
  __syncthreads();

  if (threadIdx % 2)
  {
    unsigned int group_idx;

    coalesced_group cg = coalesced_threads();
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
    }

    if (cg.num_threads() > 1) // we need at least 2 threads in the group
    {
      bool secondThread = cg.thread_rank() == 1;
      if (secondThread)
      {
        val = 777;
        laneId = threadIdx.x & 0x1f;
      }
      laneId = cg.shfl(laneId, 1);
      bool leader = cg.thread_rank() == 0;
      if (leader)
      {
        val = 777;
        auto mask = cg.match_any(val); // the mast should be equal to the mask of thread_rank 1 and thread_rank 0
        val = 0; // make sure the val is correctly recieved from the thread with thread_rank 1
        val = __shfl_sync(mask, val, laneId);
        out[group_idx] = val;
      }
    }
    else
    {
      out[group_idx] = -1; // the coalesced group did not meet conditions requried for test
    }
  }
}

__global__ void match_all_test(int *ng /*out int*/, int *out /*out array*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int val;
  int laneId;
  __syncthreads();

  if (threadIdx % 2)
  {
    unsigned int group_idx;

    coalesced_group cg = coalesced_threads();
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
    }
    if (cg.num_threads() > 1) // we need at least 2 threads in the group
    {
      bool secondThread = cg.thread_rank() == 1;
      if (secondThread)
      {
        val = 1;
        laneId = threadIdx.x & 0x1f;
      }
      if (cg.thread_rank() > 1) // handle cases where there are more than 2 threads in cg
      {
        val = 1;
      }

      laneId = cg.shfl(laneId, 1);
      if (leader)
      {
        val = 1; // depending on this being commented match_all can be tested for returning the mask or returning 0
        auto mask = cg.match_all(val); // = 0 || if val=1 is set in this branch mask != 0
        if (mask != 0)
        {
          val = __shfl_sync(mask, val, laneId); // all match returns mask
          out[group_idx] = val;
        }
        else
        {
          out[group_idx] = 0; // not all match returns 0 (negative)
        }
      }
    }
    else
    {
      out[group_idx] = -1; // the coalesced group did not meet conditions requried for test
    }
  }
}

__global__ void labeled_partition_test()
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int val = 0;
  /// The following code will create a 32-thread tile
  thread_block block = this_thread_block();
  thread_block_tile<32> tile32 = tiled_partition<32>(block);
  __syncthreads();

  if (threadIdx % 2)
  {
    val = 5;
    __syncthreads();
  }
  // the groups created here should be exactly 16 threads/lanes/warp items/flux capacitors(why not?)
  coalesced_group cg = labeled_partition(tile32, val);
}

__global__ void binary_partition_test()
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val = false;
  /// The following code will create a 32-thread tile
  thread_block block = this_thread_block();
  thread_block_tile<32> tile32 = tiled_partition<32>(block);
  __syncthreads();

  if (threadIdx % 2)
  {
    val = true;
    __syncthreads();
  }
  // the groups created here should be exactly 16 threads/lanes/warp items/flux capacitors(why not?)
  coalesced_group cg = labeled_partition(tile32, val);
}

namespace coopg = cooperative_groups;

// Using wait()
// Note: kernel assumes execution with 1 warp
__global__ void memcpy_asyc_test(int *ng /*out int*/, int *out /*out array*/, int *in_data /*in array[64]*/)
{
  auto threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t = numElements = 64;
  const size_t = elemInShared = 32;
  __shared__ int shared_data[elemInShared];
  __syncthreads();
  if (threadIdx % 2)
  {
    // create thread group
    coalesced_group cg = coalesced_threads();

    unsigned int group_idx;
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      // keep track of num of groups created
      group_idx = atomicInc(&ng, UINT_MAX);
    }
    size_t data_copied = 0;
    size_t index_reached = 0;
    while (index_reached < numElements)
    {
      // copy part of the data which is large enough to fit in shared memory
      coopg::memcpy_async(cg, shared_data, elemInShared, in_data + index_reached, numElements - index_reached);

      // find how many elements have been copied
      data_copied = min(elemInShared, numElements - index_reached);
      // wait for memcpy to finish
      coopg::wait(cg);
      if (leader)
      {
        int sum = 0;
        for (size_t i = 0; i < numElements; i++)
        {
          sum += shared_data_final[i];
        }
        // sum the data which has been copied into shared memory
        out[group_idx] += sum;
      }
      // offset the index and repeat loop until all data is transfered
      index_reached += data_copied;
    }
  }
}

__global__ void reduce_test(int *ng /*out int*/, int *out /*out array*/, int *in_data /*in array[64]*/)
{
  size_t threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t = numElements = 64;

  __syncthreads();
  if (threadIdx % 2)
  {
    coalesced_group cg = coalesced_threads();

    unsigned int group_idx;
    bool leader = cg.thread_rank() == 0;
    if (leader)
    {
      group_idx = atomicInc(&ng, UINT_MAX);
    }
    int sum = coopg::reduce(cg, in_data[cg.thread_rank()], cg::plus<int>()); // we assume data in is [1,1,...,1] so out[ng] = ng;
    // other variations
    /*
    //cg::less
    int min = coopg::reduce(cg, in_data[cg.thread_rank()], cg::less<int>()); // we assume data in is [1,2,3,...,64] so out[ng] = 1;
    //cg::greater
    int max = coopg::reduce(cg, in_data[numElements - cg.thread_rank()-1], cg::greater<int>()); // we assume data in is [1,2,3...,64] so out[ng] = 64;
    //cg::bit_and
    int and = coopg::reduce(cg, in_data[cg.thread_rank()], cg::bit_and<int>()); // we assume data in is [1,1,...,1] so out[ng] = 1;
    //cg::bit_xor
    int xor = coopg::reduce(cg, in_data[cg.thread_rank()], cg::bit_xor<int>()); // we assume data in is [1,1,...,1] so out[ng] = 0;
    //cg::bit_or
    int or = coopg::reduce(cg, in_data[cg.thread_rank()], cg::bit_or<int>()); // we assume data in is [1,1,...,1] so out[ng] = 1;
    */
    if (leader)
    {
      out[group_idx] = sum;
    }
  }
}
