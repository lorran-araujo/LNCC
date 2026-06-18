lib:
	gcc -c index.c -o index.o

main:
	gcc main.c index.o -o ./bin/main
	./bin/main

deb:
	gcc -g main.c index.o -o ./bin/main
	gcc -g teste.c index.o -o ./bin/teste

teste:
	gcc teste.c index.o -o ./bin/teste
	./bin/teste

rm:
	rm ./*.o ./bin/*

valteste:
	gcc -g teste.c index.o -o ./bin/teste
	valgrind ./bin/teste

valmain:
	gcc -g main.c index.o -o ./bin/main
	valgrind ./bin/main


