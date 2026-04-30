#include "NemlApp.h"
#include "MooseMain.h"

int
main(int argc, char * argv[])
{
  return Moose::main<NemlApp>(argc, argv);
}
