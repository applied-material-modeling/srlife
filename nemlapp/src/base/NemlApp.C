#include "NemlApp.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
NemlApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

NemlApp::NemlApp(InputParameters parameters) : MooseApp(parameters)
{
  NemlApp::registerAll(_factory, _action_factory, _syntax);
}

void
NemlApp::registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  ModulesApp::registerAllObjects<NemlApp>(f, af, s);
  Registry::registerObjectsTo(f, {"NemlApp"});
  Registry::registerActionsTo(af, {"NemlApp"});
}

void
NemlApp::registerApps()
{
  registerApp(NemlApp);
}

extern "C" void
NemlApp__registerApps()
{
  NemlApp::registerApps();
}

extern "C" void
NemlApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  NemlApp::registerAll(f, af, s);
}
