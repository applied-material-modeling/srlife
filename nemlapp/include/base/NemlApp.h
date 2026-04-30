#pragma once

#include "MooseApp.h"

class NemlApp : public MooseApp
{
public:
  static InputParameters validParams();
  NemlApp(InputParameters parameters);

  static void registerApps();
  static void registerAll(Factory & f, ActionFactory & af, Syntax & s);
};
