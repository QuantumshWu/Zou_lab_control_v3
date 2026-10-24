// Compatibility entry: all native dependencies now use the shared pinned cache.
import {prepareNativeDependencies} from './prepare-native-deps.mjs';
const paths=await prepareNativeDependencies();
console.log(JSON.stringify({source:paths.superluSource,lockSha256:paths.lockSha256}));
