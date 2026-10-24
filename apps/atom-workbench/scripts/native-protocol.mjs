import {EFFECTIVE_MODEL_SCHEMA as ATOMIC_MODEL_SCHEMA,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';

export function nativeCapabilities(message){
  if(message?.id!==0||message.error||message.protocol!==NATIVE_PROTOCOL||!Array.isArray(message.capabilities)||
    message.capabilities.some(value=>typeof value!=='string')||!message.capabilities.includes(ATOMIC_MODEL_SCHEMA)){
    throw new Error(`Rust 求解器不支持 ${ATOMIC_MODEL_SCHEMA} 辐射模型；需要 ${NATIVE_PROTOCOL}，不能使用旧衰变通道代算。`);
  }
  return {protocol:message.protocol,capabilities:[...new Set(message.capabilities)]};
}

export function validateAtomicResult(result,request){
  if(request.modelSchema!==ATOMIC_MODEL_SCHEMA||request.model?.dimension!==32||!Array.isArray(request.model?.collapse))throw new Error('Atomic request is missing required physical model metadata.');
  if(result?.modelSchema!==request.modelSchema||result.protocol!==NATIVE_PROTOCOL)throw new Error('Native atomic response has missing or incompatible radiation model identity.');
}

/** Timing/sample options do not change the long-time state. Every physical
 * operator and the complete radiation envelope do, and are retained exactly. */
export function physicalRequestIdentity(request,protocol){
  return {protocol,modelSchema:request.modelSchema,radiation:request.radiation,model:request.model};
}

