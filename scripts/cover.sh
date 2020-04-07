#! /usr/bin/bash

echo "test start"

lcov -z -d ./
./dig data/big.g 0.1
gcov -a -b -c dig.cpp
lcov --no-external --directory . --capture --output-file dig_big.info
#genhtml --output-directory . --frames --show-details dig_big.info
#mv 4.8.2 RESULT/big
#mv dig.cpp.gcov RESULT/big/

echo "big data tested"

lcov -z -d ./
./dig data/Chemical_340 0.1
gcov -a -b -c dig.cpp
lcov --no-external --directory . --capture --output-file dig_chemical.info
#genhtml --output-directory . --frames --show-details dig_chemical.info
#mv 4.8.2 RESULT/chemical
#mv dig.cpp.gcov RESULT/chemical/

echo "chemical data tested"

lcov -z -d ./
./dig data/Compound_422 0.1
gcov -a -b -c dig.cpp
lcov --no-external --directory . --capture --output-file dig_compound.info
#genhtml --output-directory . --frames --show-details dig_compound.info
#mv 4.8.2 RESULT/compound
#mv dig.cpp.gcov RESULT/compound/

echo "compound data tested"

lcov -add-tracefile dig.info -a dig_big.info -a dig_chemical.info -a dig_compound.info
genhtml --output-directory RESULT --frames --show-details dig.info

