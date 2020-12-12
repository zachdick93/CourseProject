
LINK_TARGET = prog.exe
OBJS = main.o Algebra.o LRR_Model.o LRR.o RatingRegression.o SpaVector.o Vector4Review.o Utilities.o LBFGS.o Mcsrch.o
REBUILDABLES = $(OBJS) $(LINK_TARGET)

vpath %.cpp src
vpath %.cpp src/algebra
vpath %.cpp src/lara
vpath %.cpp src/optimization
vpath %.cpp src/Utilities
vpath %.hpp include/algebra
vpath %.hpp include/lara
vpath %.hpp include/optimization
vpath %.hpp include/Utilities

clean : 
	rm -f $(REBUILDABLES)
	echo "Clean Done"

all : $(LINK_TARGET)

$(LINK_TARGET) : $(OBJS)
	g++ -g -o $@ $^

%.o : %.cpp
	g++ -g -o $@ -c $<

Main.o : LRR.hpp RatingRegression.hpp

Algebra.o : Algebra.hpp

LRR_Model.o : LRR_Model.hpp Algebra.hpp Utilities.hpp

LRR.o : LRR.hpp LRR_Model.hpp RatingRegression.hpp LBFGS.hpp Utilities.hpp

RatingRegression.o : RatingRegression.hpp SpaVector.hpp LBFGS.hpp Utilities.hpp

SpaVector.o : SpaVector.hpp

Vector4Review.o : Vector4Review.hpp SpaVector.hpp RatingRegression.hpp LRR.hpp Utilities.hpp

LBFGS.o : LBFGS.hpp Mcsrch.hpp

Mcsrch.o : Mcsrch.hpp LBFGS.hpp

Utilities.o : Utilities.hpp
