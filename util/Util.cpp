/*=============================================================================
# Filename: Util.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 17:23
# Description: 
=============================================================================*/

#include "Util.h"

using namespace std;

Util::Util()
{
}

Util::~Util()
{
}

long
Util::get_cur_time()
{
    timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000 + tv.tv_usec/1000);
}

unsigned 
Util::RoundUp(int num, int base)
{
    unsigned tmp = (num+base-1)/base;
    return tmp * base;
}

unsigned 
Util::RoundUpDivision(int num, int base)
{
    return (num+base-1)/base;
}

//the seed is a prime, which can be well chosed to yield good performance(low conflicts)
uint32_t 
Util::MurmurHash2(const void * key, int len, uint32_t seed) 
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.
    const uint32_t m = 0x5bd1e995;
    const int r = 24;
    // Initialize the hash to a 'random' value
    uint32_t h = seed ^ len;
    // Mix 4 bytes at a time into the hash
    const unsigned char * data = (const unsigned char *) key;
    while (len >= 4) 
    {
        uint32_t k = *(uint32_t*) data;
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        data += 4;
        len -= 4;
    }
    // Handle the last few bytes of the input array
    switch (len) 
    {
        case 3:
            h ^= data[2] << 16;
        case 2:
            h ^= data[1] << 8;
        case 1:
          h ^= data[0];
          h *= m;
    };
    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}

void 
Util::DisplayBinary(int num)
{
    int i, j;
    j = sizeof(int)*8;
    for (i = 0; i < j; i ++)
    {
        //NOTICE:the most significant bit is output first
        //&0x1 is needed because for negative integers the right shift will add 1 in the former bits
        printf("%d", (num>>(j-i-1)&0x1));
    }
}

