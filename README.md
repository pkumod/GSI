# GSI

GPU-friendly Subgraph Isomorphism 

We target at a one-to-one mapping at a time, the query graph is small(vertices less than 100), while the data graph can be very large.
(but all can be placed in GPU's global memory)

---

####  Dataset

NOTICE: we add 1 to labels for both vertex and edge, to ensure the label is positive!

see `data/readme` and `data/*.g`

---

#### Publication

Li Zeng, Lei Zou, M. Tamer Özsu, Lin Hu, Fan Zhang. **GSI: GPU-friendly Subgraph Isomorphism**. *International Conference on Data Engineering*, 2020.

---

#### Others

My implementation of [GunrockSM](https://github.com/bookug/GunrockSM), [GpSM](https://github.com/bookug/GpSM), and [GPU utilities](https://github.com/bookug/gutil).

