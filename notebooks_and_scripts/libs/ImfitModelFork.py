#! /usr/bin/env python
import numpy as np

def parse_imfit_line(line):
    """ Function parses line of imfit data file and
    returns parameters"""
    params = line.split()
    name = params[0]
    value = float(params[1])  # Value of the parameter is the second entry of the line
    # (after the name)
    # Lets now find range of values. Its have to contain the comma and must
    # be the second enrty (otherwise imfit wont work).
    # Some parameters can be fixed, so we have to check this possibility at first
    if (len(params) == 2) or ("fixed" in params[2]):
        # No bounds specified at all or fixed value
        lowerLim = upperLim = None
    else:
        rangeParams = params[2].split(",")
        lowerLim = float(rangeParams[0])
        upperLim = float(rangeParams[1])
    fixed = False
    try:
        if ("fixed" in params[2]):
            fixed = True
    except IndexError:
        pass

    return ImfitParameter(name, value, lowerLim, upperLim, fixed)

class ImfitParameter(object):
    """ Just a container of parameter instance:
    parameter name, value and its range"""
    def __init__(self, name, value, lowerLim, upperLim, fixed):
        self.name = name
        self.value = value
        self.lowerLim = lowerLim
        self.upperLim = upperLim
        self.fixed = fixed
        self.badBoundary = False

class ImfitFunction(object):
    """ Class represents imfit function with
    all its parameters their ranges"""
    def __init__(self, funcName, ident, comment = None):
        # ident is a unical number of the function
        # funcName is just a type of the function and it can be
        # the same for different galaxy components
        self.name = funcName
        self.ident = ident
        self.uname = "%s.%i" % (funcName, ident)  # func unique name
        self.params = []
        self.comment = comment

    def add_parameter(self, newParameter):
        self.params.append(newParameter)

    def num_of_params(self):
        return len(self.params)

    def get_par_by_name(self, name):
        for par in self.params:
            if par.name == name:
                return par

    def get_size(self):
        """ Return a parameter that corresponds to the characteristic size of the component """
        if self.name in ("Exponential", "ExponentialDisk3D", "EdgeOnDisk", "Exponential_GenEllipse"):
            return self.get_par_by_name("h").value
        if self.name in ("Sersic", "Core-Sersic", "Sersic_GenEllipse"):
            return self.get_par_by_name("r_e").value
        if self.name in ("BrokenExponential2D", "BrokenExponentialDisk3D", "BknExp3D",
                         "BrokenExponential", "BrokenExponential2"):
            return 0.5 * (self.get_par_by_name("h1").value + self.get_par_by_name("h2").value)
        if self.name in ("DblBknExp3D", "DoubleBrokenExponential"):
            return (self.get_par_by_name("h1").value + self.get_par_by_name("h2").value +
                    self.get_par_by_name("h3").value) / 3
        return None

    def get_center(self):
        return self.get_par_by_name("X0").value, self.get_par_by_name("Y0").value


class ImfitModel(object):
    """Imfit functions and their parameters"""
    def __init__(self, modelFileName):
        #print("Reading '%s':" % (modelFileName))
        # Read imfit input file
        self.listOfFunctions = []
        self.numberOfParams = 0
        funcName = None
        currentFunction = None
        ident = -1
        for line in open(modelFileName):
            sLine = line.strip()
            if sLine.startswith("#"):
                # It is a comment line, just skip it
                continue
            if len(sLine) == 0:
                "Empty line"
                continue
            if "#" in sLine:
                # Drop the comment part of the line if exists
                sLine = sLine[:sLine.index("#")].strip()
            if sLine.startswith("X0"):
                x0 = parse_imfit_line(sLine)
            elif sLine.startswith("Y0"):
                y0 = parse_imfit_line(sLine)
            elif sLine.startswith("FUNCTION"):
                # New function is found.
                ident += 1
                # If we are working already with some function, then
                # the list of parameters for this function is over and we can
                # add it to the function list
                if funcName is not None:
                    self.listOfFunctions.append(currentFunction)
                funcName = sLine.split()[1]
                currentFunction = ImfitFunction(funcName, ident)
                currentFunction.add_parameter(x0)
                currentFunction.add_parameter(y0)
                self.numberOfParams += 2
            else:
                # If line does not contain nor coordinates nor function name
                # then in has to be a parameter line
                param = parse_imfit_line(sLine)
                currentFunction.add_parameter(param)
                self.numberOfParams += 1
        # append the last function
        self.listOfFunctions.append(currentFunction)
        # Print some statistics
        #print("  %i functions found (%i parameters)\n" % (len(self.listOfFunctions), self.numberOfParams))

    def get_func_by_uname(self, uname):
        for func in self.listOfFunctions:
            if uname == func.uname:
                return func

    def get_disc(self):
        """
        Function returns disc component.
        """
        exps = []
        for func in self.listOfFunctions:
            if func.name in ['Exponential', 'BrokenExponential', 'Exponential_GenEllipse']:
                exps.append(func)
        if len(exps) == 0:
            exps = self.listOfFunctions
        sizes = np.array([component.get_size() for component in exps], dtype=float)
        return exps[np.argsort(1/sizes)[0]]