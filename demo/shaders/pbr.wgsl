let MONTECARLO_NUM_SAMPLES = 2048;
let PI = 3.1415998935699463;

struct MaterialInfo {
    baseColor: vec3<f32>,
    specularColor0_: vec3<f32>,
    specularColor90_: vec3<f32>,
    perceptualRoughness: f32,
    metallic: f32,
    ior: f32,
    diffuseColor: vec3<f32>,
    specularWeight: f32,
}

struct LightingGeneralParams {
    surfaceNormal: vec3<f32>,
    viewDirection: vec3<f32>,
}

struct LightingVectorSample {
    pdf: f32,
    localDirection: vec3<f32>,
}

// Hammersley Points on the Hemisphere
// CC BY 3.0 (Holger Dammertz)
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// with adapted interface
fn radicalInverse_VdC(bits:u32) -> f32 {
    var bits = bits;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// hammersley2d describes a sequence of points in the 2d unit square [0,1)^2
// that can be used for quasi Monte Carlo integration
fn hammersley2d(i: i32, N: i32) -> vec2<f32> {
    return vec2<f32>(f32(i)/f32(N), radicalInverse_VdC(u32(i)));
}


fn localHemisphereDirectionHelper(cosTheta: f32, sinTheta: f32, phi: f32) -> vec3<f32> {
	return vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

fn max3_(v: vec3<f32>) -> f32 {
  return max(max(v.x, v.y), v.z);
}

fn normalDistributionLambert(nDotH: f32) -> f32 {
  return PI;
}

fn importanceSampleLambert(uv: vec2<f32>) -> LightingVectorSample {
  var result: LightingVectorSample;
	let phi = uv.x * 2.0 * PI;
	let cosTheta = sqrt(1.0 - uv.y);
	let sinTheta = sqrt(uv.y);

	result.pdf = cosTheta / PI;
	result.localDirection = localHemisphereDirectionHelper(cosTheta, sinTheta, phi);
	return result;
}

fn normalDistributionGGX(nDotH: f32, roughness: f32) -> f32 {
	let roughnessSq = roughness * roughness;
	let f = (nDotH * nDotH) * (roughnessSq - 1.0) + 1.0;
	return roughnessSq / (PI * f * f);
}

fn importanceSampleGGX(uv: vec2<f32>, roughness: f32) -> LightingVectorSample {
	var result: LightingVectorSample;
	let roughnessSq = roughness * roughness;
	let phi = 2.0 * PI * uv.x;
	let cosTheta = sqrt((1.0 - uv.y) / (1.0 + (roughnessSq * roughnessSq - 1.0) * uv.y));
	let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	result.pdf = normalDistributionGGX(cosTheta, roughness * roughness);
	result.localDirection = localHemisphereDirectionHelper(cosTheta, sinTheta, phi);
	return result;
}

fn visibilityGGX(nDotL: f32, nDotV: f32, roughness: f32) -> f32 {
	let roughnessSq = roughness * roughness;
	let GGXV = nDotL * sqrt(nDotV * nDotV * (1.0 - roughnessSq) + roughnessSq);
	let GGXL = nDotV * sqrt(nDotL * nDotL * (1.0 - roughnessSq) + roughnessSq);
	return 0.5 / (GGXV + GGXL);
}

fn normalDistributionCharlie(sheenRoughness: f32, NdotH: f32) -> f32 {
	let sheenRoughness = max(sheenRoughness, 0.000001); //clamp (0,1]
	let alphaG = sheenRoughness * sheenRoughness;
	let invR = 1.0 / alphaG;
	let cos2h = NdotH * NdotH;
	let sin2h = 1.0 - cos2h;
	return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

fn lambdaSheenNumericHelper(x: f32, alphaG: f32) -> f32 {
	let oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
	let a = mix(21.5473, 25.3245, oneMinusAlphaSq);
	let b = mix(3.82987, 3.32435, oneMinusAlphaSq);
	let c = mix(0.19823, 0.16801, oneMinusAlphaSq);
	let d = mix(-1.97760, -1.27393, oneMinusAlphaSq);
	let e = mix(-4.32054, -4.85967, oneMinusAlphaSq);
	return a / (1.0 + b * pow(x, c)) + d * x + e;
}

fn lambdaSheen(cosTheta: f32, alphaG: f32) -> f32 {
	if (abs(cosTheta) < 0.5) {
		return exp(lambdaSheenNumericHelper(cosTheta, alphaG));
	}
	else {
		return exp(2.0 * lambdaSheenNumericHelper(0.5, alphaG) - lambdaSheenNumericHelper(1.0 - cosTheta, alphaG));
	}
}

fn visiblitySheen(nDotL: f32, nDotV: f32, sheenRoughness: f32) -> f32 {
	let sheenRoughness = max(sheenRoughness, 0.000001); //clamp (0,1]
	let alphaG = sheenRoughness * sheenRoughness;

	return clamp(1.0 / ((1.0 + lambdaSheen(nDotV, alphaG) + lambdaSheen(nDotL, alphaG)) * (4.0 * nDotV * nDotL)), 0.0, 1.0);
}

fn fresnelSchlick(f0: vec3<f32>, f90: vec3<f32>, vDotH: f32) -> vec3<f32> {
	return f0 + (f90 - f0) * pow(clamp(1.0 - vDotH, 0.0, 1.0), 5.0);
}

fn getIBLRadianceGGX(ggxSample: vec3<f32>, ggxLUT: vec2<f32>, fresnelColor: vec3<f32>, specularWeight: f32) -> vec3<f32> {
  let FssEss = fresnelColor * ggxLUT.x + ggxLUT.y;
	return specularWeight * ggxSample * FssEss;
}

fn getIBLRadianceLambertian(lambertianSample: vec3<f32>, ggxLUT: vec2<f32>, fresnelColor: vec3<f32>, diffuseColor: vec3<f32>, F0: vec3<f32>, specularWeight: f32) -> vec3<f32> {
  let FssEss = specularWeight * fresnelColor * ggxLUT.x + ggxLUT.y; // <--- GGX / specular light contribution (scale it down if the specularWeight is low)

	// Multiple scattering, from Fdez-Aguera
	let Ems = (1.0 - (ggxLUT.x + ggxLUT.y));
	let F_avg = specularWeight * (F0 + (1.0 - F0) / 21.0);
	let FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
	let k_D = diffuseColor * (1.0 - FssEss + FmsEms); // we use +FmsEms as indicated by the formula in the blog post (might be a typo in the implementation)

	return (FmsEms + k_D) * lambertianSample;
}

fn computeEnvironmentLighting(material: MaterialInfo, params: LightingGeneralParams) -> vec3<f32> {
  return vec3<f32>();
}

fn computeLUT(in: vec2<f32>) -> vec2<f32> {
  let roughness = in.x;
  let nDotV = in.y;

  var localViewDir = vec3<f32>(0.0);
  localViewDir.x = sqrt(1.0 - nDotV * nDotV); // = sin(acos(nDotV));
  localViewDir.y = 0.0;
  localViewDir.z = nDotV;

  var result = vec2<f32>();
  var sampleIndex = 0;
  loop {
    if(sampleIndex >= MONTECARLO_NUM_SAMPLES) {
      break;
    }

    let coord = hammersley2d(sampleIndex, MONTECARLO_NUM_SAMPLES);

    let lvs = importanceSampleGGX(coord, roughness);
    let sampleNormal = lvs.localDirection;
    let localLightDir = reflect(localViewDir, sampleNormal);

    let nDotL = localLightDir.z;
    let nDotH = sampleNormal.z;
    let vDotH = dot(localViewDir, sampleNormal);
    
    if(nDotL > 0.0) {
      let pdf = visibilityGGX(nDotL, nDotV, roughness) * vDotH * nDotL / nDotH;
      let fc = pow(1.0 - vDotH, 5.0);
      result.x = result.x + (1.0 - fc) * pdf;
      result.y = result.y + fc * pdf;
    }

    sampleIndex = sampleIndex + 1;
  }
  
  return result.xy / f32(MONTECARLO_NUM_SAMPLES) * 4.0;
}