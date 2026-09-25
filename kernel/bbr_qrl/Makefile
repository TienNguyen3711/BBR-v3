# Composite lab module. Generate tcp_bbr_qrl.c from the pinned source first.
obj-m += bbr_qrl.o
bbr_qrl-y := tcp_bbr_qrl.o bbr_qrl_ctl.o
KDIR ?= /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules
clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
.PHONY: all clean
