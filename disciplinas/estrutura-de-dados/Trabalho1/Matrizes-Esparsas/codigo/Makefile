lib:
	gcc -c matrix.c -o matrix.o

main:
	gcc main.c matrix.o -o ./bin/main
	./bin/main

deb:
	gcc -g main.c matrix.o -o ./bin/main
	gcc -g teste.c matrix.o -o ./bin/teste

teste:
	gcc teste.c matrix.o -o ./bin/teste
	./bin/teste

rm:
	rm ./*.o ./bin/*

valteste:
	gcc -g teste.c matrix.o -o ./bin/teste
	valgrind ./bin/teste

valmain:
	gcc -g main.c matrix.o -o ./bin/main
	valgrind ./bin/main


