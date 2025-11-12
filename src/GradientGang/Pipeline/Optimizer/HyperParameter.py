import optuna

class HyperParameter:
        # construction
        counter = 0
        def __init__(self, params, constr_map):
            self.id = str(HyperParameter.counter)
            self.name = params["name"]
            self.type = params["type"]
            self.opts = params["opts"]
            self.path = params["path"]

            # creates internal hyperparameters and constraints for each sub-architecture 
            if self.type == "arch":
                self.subparams, self.constraint = self.findSubparams(constr_map, **self.opts)
            elif self.type == "constrained":
                self.value = None
                self.valid = False

            HyperParameter.counter += 1
        
        # create hyperparameters for each sub-architecture
        def findSubparams(self, constr_map: dict, archs: dict, min_layers: int, max_layers: int, constraint: str = None):
            subs = []
            constr = None

            for k in archs:
                for h in archs[k]["hyper_arch"]:
                    for i in range(max_layers):
                        subs.append(HyperParameter(archs[k]["hyper_arch"][h], constr_map))
                        if isinstance(subs[-1].path[-1], int):
                            subs[-1].path[-1] += i
                        else:
                            subs[-1].path.append(i)
            
            if constraint != None:
                constr = HyperParameterConstraint(constr_map[constraint])
                constr.params = subs
            
            return subs, constr

        # evaluation
        def getValue(self, trial: optuna.trial.BaseTrial):
            value = None

            if not self.active:
                pass
            elif self.type == "categ":
                value = trial.suggest_categorical(self.id, **self.opts)
            elif self.type == "float":
                value = trial.suggest_float(self.id, **self.opts)
            elif self.type == "int":
                value = trial.suggest_int(self.id, **self.opts)
            elif self.type == "value":
                value = self.opts["value"]
            elif self.type == "arch":
                value = self.getArch(trial, **self.opts)

            self.active = False

            return value

        def getArch(self, trial: optuna.trial.BaseTrial, archs: dict, min_layers, max_layers, constraints):
            # choose architecture and number of layers
            name = trial.suggest_categorical(self.id + "_arch", list(archs.keys()))
            layers = trial.suggest_int(self.id + "_layers", min_layers[name], max_layers[name])

            # build architecture
            arch = [archs[name].copy() for _ in range(layers)]
            
            # call hyperparameters constraint
            if constraints != None:
                self.constraint.constraint()
            
            # build architecture
            for h in self.subparams:
                hp = self.subparams[h]
                curr = arch
                for k in hp.path[:-1]:
                    curr = curr[k]
            curr[hp.name] = hp.getValue(trial)
                
            return arch

class HyperParameterConstraint:
     def __init__(self, function):
        self.params = []
        self.constraint = function