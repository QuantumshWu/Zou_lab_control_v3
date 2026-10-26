import {FULL_MODEL_SCHEMA,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {DEFAULT_DYNAMICS_SAMPLES,resolvePopulationTolerance} from '../dist/js/atom-workbench/src/types.js';

const requestError=(message,code)=>Object.assign(new Error(message),{status:400,code});

export function nativeCapabilities(message){
  if(message?.id!==0||message.error||message.protocol!==NATIVE_PROTOCOL||!Array.isArray(message.capabilities)||
    message.capabilities.some(value=>typeof value!=='string')||!message.capabilities.includes(FULL_MODEL_SCHEMA)){
    throw new Error(`Rust 求解器不支持 ${FULL_MODEL_SCHEMA} 模型；需要 ${NATIVE_PROTOCOL}。`);
  }
  return {protocol:message.protocol,capabilities:[...new Set(message.capabilities)]};
}

/** The one mapping from an HTTP/IPC solve request (scene + options) to the
 * native payload, and from the native result back to the client response. */
export function prepareNativeSolve(input){
  if(input.modelSchema!==undefined&&input.modelSchema!==FULL_MODEL_SCHEMA)throw requestError('Unsupported atomic model schema.','incompatible-model-schema');
  const settings={...input.scene.dynamics,...input.options};
  const populationTolerance=resolvePopulationTolerance(settings.populationTolerance);
  const started=performance.now(),model=buildFullMasterModel(input.scene,settings),modelBuildMs=performance.now()-started;
  const options={durationUs:settings.durationUs??5,samples:settings.samples??DEFAULT_DYNAMICS_SAMPLES,populationTolerance,
    includeAsymptotic:input.options?.includeAsymptotic??true,steadyOnly:input.options?.steadyOnly??false};
  if(typeof options.steadyOnly!=='boolean'||typeof options.includeAsymptotic!=='boolean')throw requestError('Invalid solver options','invalid-options');
  const payload={modelSchema:model.modelSchema,options,model:{dimension:model.dimension,states:model.states.map(state=>state.id),
    hReal:model.hReal,hImag:model.hImag,collapse:model.collapse,initialIndex:model.initialIndex,drives:model.drives,frequencyCoordinates:model.frequencyCoordinates}};
  // The response carries the full level metadata in place of the native state ids.
  const respond=(result,request)=>({...result,request,states:model.states,assumptions:model.assumptions,service:{...result.service,modelBuildMs}});
  return {payload,respond};
}

export function validateAtomicResult(result,request){
  if(request.modelSchema!==FULL_MODEL_SCHEMA||request.model?.dimension!==32||!Array.isArray(request.model?.collapse))throw new Error('Atomic request is missing required physical model metadata.');
  if(result?.modelSchema!==request.modelSchema||result.protocol!==NATIVE_PROTOCOL)throw new Error('Native atomic response has missing or incompatible model identity.');
}
