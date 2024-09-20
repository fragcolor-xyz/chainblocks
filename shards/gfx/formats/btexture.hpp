#ifndef C8C54D29_755D_4D95_87CC_9D6F10462CFA
#define C8C54D29_755D_4D95_87CC_9D6F10462CFA

#include <boost/uuid/uuid.hpp>

namespace gfx {

struct BTextureMetadata {
  boost::uuids::uuid uuid;
  std::string path;

};

struct BTextureRef {
  boost::uuids::uuid uuid;
};
} // namespace gfx

#endif /* C8C54D29_755D_4D95_87CC_9D6F10462CFA */
