#ifndef SQLITE_VEC_H
#define SQLITE_VEC_H

#include "sqlite3ext.h"

#ifdef SQLITE_VEC_STATIC
#define SQLITE_VEC_API
#else
#ifdef _WIN32
#define SQLITE_VEC_API __declspec(dllexport)
#else
#define SQLITE_VEC_API
#endif
#endif

#define SQLITE_VEC_VERSION "v0.1shards"
#define SQLITE_VEC_DATE ""
#define SQLITE_VEC_SOURCE ""

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg,
                  const sqlite3_api_routines *pApi);

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_vec_fs_read_init(sqlite3 *db, char **pzErrMsg,
                          const sqlite3_api_routines *pApi);

#ifdef __cplusplus
} /* end of the 'extern "C"' block */
#endif

#endif /* ifndef SQLITE_VEC_H */
