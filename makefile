#此行为注释
#main: main.o Circle.o 
#	g++ main.o Circle.o -o main

main:main.o allocation.o arena_planner.o context.o error_reporter.o exception_jni.o \
	graph_info.o interpreter.o model.o nativeinterpreterwrapper_jni.o nnapi_delegate.o \
	simple_memory_arena.o tf_util.o DataType.o
	g++ main.o allocation.o arena_planner.o context.o error_reporter.o exception_jni.o \
	graph_info.o interpreter.o model.o nativeinterpreterwrapper_jni.o nnapi_delegate.o \
	simple_memory_arena.o tf_util.o DataType.o -o main  `pkg-config --libs --cflags opencv` -ldl

allocation.o:allocation.cc
	g++ -std=gnu++0x -c allocation.cc -o allocation.o

arena_planner.o:arena_planner.cc
	g++ -std=gnu++0x -c arena_planner.cc -o arena_planner.o

context.o:context.c
	g++ -std=gnu++0x -c context.c -o context.o

DataType.cc:DataType.o
	g++ -std=gnu++0x -c DataType.cc -o DataType.o

error_reporter.o:error_reporter.cc
	g++ -std=gnu++0x -c error_reporter.cc -o error_reporter.o

exception_jni.o:exception_jni.cc
	g++ -std=gnu++0x -c exception_jni.cc -o exception_jni.o

graph_info.o:graph_info.cc
	g++ -std=gnu++0x -c graph_info.cc -o graph_info.o

interpreter.o:interpreter.cc
	g++ -std=gnu++0x -c interpreter.cc -o interpreter.o

model.o:model.cc
	g++ -std=gnu++0x -c model.cc -o model.o

nativeinterpreterwrapper_jni.o:nativeinterpreterwrapper_jni.cc
	g++ -std=gnu++0x -c nativeinterpreterwrapper_jni.cc -o nativeinterpreterwrapper_jni.o

nnapi_delegate.o:nnapi_delegate.cc
	g++ -std=gnu++0x -c nnapi_delegate.cc -o nnapi_delegate.o

simple_memory_arena.o:simple_memory_arena.cc
	g++ -std=gnu++0x -c simple_memory_arena.cc -o simple_memory_arena.o

tf_util.o:tf_util.cc
	g++ -std=gnu++0x -c tf_util.cc -o tf_util.o

main.o:main.cc
	g++ -std=gnu++0x -c main.cc -o main.o

clean:
	rm -r *.o
