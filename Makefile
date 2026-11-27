.PHONY: all clean install uninstall

all:
	$(MAKE) -C cpp_stacker

install:
	$(MAKE) -C cpp_stacker install

uninstall:
	$(MAKE) -C cpp_stacker uninstall

clean:
	$(MAKE) -C cpp_stacker clean
