class JaniSyntaxError(SyntaxError):
    pass

class UnsupportedFeatureError(Exception):
    pass

class RequiredConstantExpressionError(Exception):
    pass

class MissingModelParameterError(Exception):
    pass

class JaniRRequirementError(SyntaxError):
    pass

class JaniRSyntaxError(JaniSyntaxError):
    pass