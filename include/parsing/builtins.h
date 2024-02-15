#ifndef KIWI_PARSING_BUILTINS_H
#define KIWI_PARSING_BUILTINS_H

#include <string>
#include <unordered_set>

struct {
  // File operations
  const std::string AppendText = "__appendtext__";
  const std::string CopyFile = "__copyfile__";
  const std::string CreateFile = "__createfile__";
  const std::string DeleteFile = "__deletefile__";
  const std::string MoveFile = "__movefile__";
  const std::string ReadFile = "__readfile__";
  const std::string ReadLines = "__readlines__";
  const std::string WriteLine = "__writeline__";
  const std::string WriteText = "__writetext__";
  const std::string FileExists = "__fileexists__";
  const std::string GetFileExtension = "__fileext__";
  const std::string FileName = "__filename__";
  const std::string FileSize = "__filesize__";
  const std::string GetFilePath = "__filepath__";
  const std::string GetFileAbsolutePath = "__fileabspath__";
  const std::string GetFileAttributes = "__fileattrs__";  // TODO
  const std::string Glob = "__glob__";                    // I <3 Ruby.

  // Directory operations
  const std::string DirectoryExists = "__direxists__";
  const std::string DirectoryName = "__dirname__";
  const std::string ListDirectory = "__listdir__";
  const std::string MakeDirectory = "__mkdir__";
  const std::string MakeDirectoryP = "__mkdirp__";
  const std::string RemoveDirectory = "__rmdir__";
  const std::string IsDirectory = "__isdir__";
  const std::string ChangeDirectory = "__chdir__";
  const std::string GetCurrentDirectory = "__cwd__";

  std::unordered_set<std::string> builtins = {AppendText,
                                              CopyFile,
                                              CreateFile,
                                              DeleteFile,
                                              MoveFile,
                                              ReadFile,
                                              ReadLines,
                                              WriteText,
                                              WriteLine,
                                              FileExists,
                                              FileName,
                                              FileSize,
                                              GetFileExtension,
                                              GetFilePath,
                                              GetFileAttributes,
                                              DirectoryExists,
                                              DirectoryName,
                                              ListDirectory,
                                              MakeDirectory,
                                              MakeDirectoryP,
                                              RemoveDirectory,
                                              IsDirectory,
                                              ChangeDirectory,
                                              GetCurrentDirectory,
                                              GetFileAbsolutePath,
                                              Glob};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} FileIOBuiltIns;

struct {
  const std::string Home = "__home__";

  std::unordered_set<std::string> builtins = {Home};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} ModuleBuiltins;

struct {
  const std::string Hour = "__hour__";
  const std::string Minute = "__min__";
  const std::string Second = "__sec__";
  const std::string MonthDay = "__mday__";
  const std::string WeekDay = "__wday__";
  const std::string YearDay = "__yday__";
  const std::string Month = "__mon__";
  const std::string Year = "__year__";
  const std::string EpochMilliseconds = "__epochms__";
  const std::string Delay = "__delay__";
  const std::string IsDST = "__isdst__";
  const std::string Ticks = "__ticks__";
  const std::string TicksToMilliseconds = "__ticksms__";
  const std::string AMPM = "__ampm__";

  std::unordered_set<std::string> builtins = {
      Hour,    Minute, Second, MonthDay,           WeekDay,
      YearDay, Month,  Year,   EpochMilliseconds,  Delay,
      IsDST,   Ticks,  AMPM,   TicksToMilliseconds};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} TimeBuiltins;

struct {
  const std::string Sin = "__sin__";
  const std::string Tan = "__tan__";
  const std::string Asin = "__asin__";
  const std::string Acos = "__acos__";
  const std::string Atan = "__atan__";
  const std::string Atan2 = "__atan2__";
  const std::string Sinh = "__sinh__";
  const std::string Cosh = "__cosh__";
  const std::string Tanh = "__tanh__";
  const std::string Cos = "__cos__";
  const std::string Log = "__log__";
  const std::string Log2 = "__log2__";
  const std::string Log10 = "__log10__";
  const std::string Log1P = "__log1p__";
  const std::string Fmod = "__fmod__";
  const std::string Hypot = "__hypot__";
  const std::string IsFinite = "__isfinite__";
  const std::string IsInf = "__isinf__";
  const std::string IsNaN = "__isnan__";
  const std::string IsNormal = "__isnormal__";
  const std::string Sqrt = "__sqrt__";
  const std::string Cbrt = "__cbrt__";
  const std::string Abs = "__abs__";
  const std::string Floor = "__floor__";
  const std::string Ceil = "__ceil__";
  const std::string Round = "__round__";
  const std::string Trunc = "__trunc__";
  const std::string Remainder = "__remainder__";
  const std::string Exp = "__exp__";
  const std::string ExpM1 = "__expm1__";
  const std::string Erf = "__erf__";
  const std::string ErfC = "__erfc__";
  const std::string LGamma = "__lgamma__";
  const std::string TGamma = "__tgamma__";
  const std::string FMax = "__fmax__";
  const std::string FMin = "__fmin__";
  const std::string FDim = "__fdim__";
  const std::string CopySign = "__copysign__";
  const std::string NextAfter = "__nextafter__";
  const std::string Pow = "__pow__";
  const std::string Epsilon = "__eps__";
  const std::string Random = "__random__";

  std::unordered_set<std::string> builtins = {
      Sin,  Tan,   Asin,     Acos,      Atan,   Atan2,    Sinh,
      Cosh, Tanh,  Cos,      Log,       Log2,   Log10,    Log1P,
      Fmod, Hypot, IsFinite, IsInf,     IsNaN,  IsNormal, Sqrt,
      Cbrt, Abs,   Floor,    Ceil,      Round,  Trunc,    Remainder,
      Exp,  ExpM1, Erf,      ErfC,      LGamma, TGamma,   FMax,
      FMin, FDim,  CopySign, NextAfter, Pow,    Epsilon,  Random};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} MathBuiltins;

struct {
  const std::string Map = "map";
  const std::string Reduce = "reduce";
  const std::string Select = "select";
  const std::string ToH = "to_h";

  std::unordered_set<std::string> builtins = {Map, Reduce, Select, ToH};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }

} SpecializedBuiltins;

#ifdef EXPERIMENTAL_FEATURES
struct {
  const std::string Get = "__http_get__";
  const std::string Post = "__http_post__";
  const std::string Put = "__http_put__";
  const std::string Delete = "__http_delete__";
  const std::string Patch = "__http_patch__";
  const std::string Head = "__http_head__";
  const std::string Options = "__http_options__";

  std::unordered_set<std::string> builtins = {Get,   Post, Put,    Delete,
                                              Patch, Head, Options};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} HttpBuiltins;

struct {
  const std::string Connect = "__odbc_connect__";
  const std::string Exec = "__odbc_exec__";
  const std::string ExecSp = "__odbc_execsp__";
  const std::string IsConnected = "__odbc_isconnected__";
  const std::string BeginTransaction = "__odbc_begin_tx__";
  const std::string CommitTransaction = "__odbc_commit_tx__";
  const std::string RollbackTransaction = "__odbc_rollback_tx__";
  const std::string InTransaction = "__odbc_in_tx__";

  std::unordered_set<std::string> builtins = {Connect,
                                              Exec,
                                              ExecSp,
                                              IsConnected,
                                              BeginTransaction,
                                              CommitTransaction,
                                              RollbackTransaction,
                                              InTransaction};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} OdbcBuiltins;
#endif

struct {
  const std::string GetEnvironmentVariable = "__getenv__";
  const std::string SetEnvironmentVariable = "__setenv__";

  std::unordered_set<std::string> builtins = {GetEnvironmentVariable,
                                              SetEnvironmentVariable};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} EnvBuiltins;

struct {
  const std::string GetArgv = "__argv__";
  const std::string GetXarg = "__xarg__";

  std::unordered_set<std::string> builtins = {GetArgv, GetXarg};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} ArgvBuiltins;

struct {
  const std::string Exec = "__exec__";
  const std::string ExecOut = "__execout__";

  std::unordered_set<std::string> builtins = {Exec, ExecOut};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} SysBuiltins;

struct {
  const std::string Input = "input";

  std::unordered_set<std::string> builtins = {Input};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }
} ConsoleBuiltins;

struct {
  const std::string Chars = "chars";
  const std::string IsA = "is_a";
  const std::string Join = "join";
  const std::string Split = "split";
  const std::string LeftTrim = "ltrim";
  const std::string RightTrim = "rtrim";
  const std::string Trim = "trim";
  const std::string Size = "size";
  const std::string ToD = "to_d";
  const std::string ToI = "to_i";
  const std::string ToS = "to_s";
  const std::string Type = "type";
  const std::string Replace = "replace";
  const std::string Contains = "contains";
  const std::string BeginsWith = "begins_with";
  const std::string EndsWith = "ends_with";
  const std::string IndexOf = "index_of";
  const std::string Upcase = "upcase";
  const std::string Downcase = "downcase";
  const std::string Keys = "keys";

  std::unordered_set<std::string> builtins = {
      Chars,    IsA,     Join,     Size,       ToD,       ToI,     ToS,
      Replace,  Replace, Contains, BeginsWith, EndsWith,  IndexOf, Upcase,
      Downcase, Keys,    Split,    LeftTrim,   RightTrim, Trim,    Type};

  bool is_builtin(const std::string& arg) {
    return builtins.find(arg) != builtins.end();
  }

  bool is_builtin_method(const std::string& arg) {
#ifdef EXPERIMENTAL_FEATURES
    return ConsoleBuiltins.is_builtin(arg) || EnvBuiltins.is_builtin(arg) ||
           ArgvBuiltins.is_builtin(arg) || TimeBuiltins.is_builtin(arg) ||
           FileIOBuiltIns.is_builtin(arg) || MathBuiltins.is_builtin(arg) ||
           ModuleBuiltins.is_builtin(arg) || SysBuiltins.is_builtin(arg) ||
           HttpBuiltins.is_builtin(arg) || OdbcBuiltins.is_builtin(arg);
#endif
    return ConsoleBuiltins.is_builtin(arg) || EnvBuiltins.is_builtin(arg) ||
           ArgvBuiltins.is_builtin(arg) || TimeBuiltins.is_builtin(arg) ||
           FileIOBuiltIns.is_builtin(arg) || MathBuiltins.is_builtin(arg) ||
           ModuleBuiltins.is_builtin(arg) || SysBuiltins.is_builtin(arg);
  }
} KiwiBuiltins;

#endif
