# GSM

GPU-friendly Subgraph Isomorphism 

we target at a one-to-one mapping at a time, the query graph is small(vertices less than 100), while the data graph can be very large.
(but all can be placed in GPU's global memory)

---

####  Dataset

NOTICE: we add 1 to labels for both vertex and edge, to ensure the label is positive!

see `data/readme` and `data/*.g`


