"""Diagnostic project gates, not a calibrated identity verification standard."""
import numpy as np


def cosine(a,b):
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))


def aperture(face):
    p=face.landmark_3d_68[:,:2]
    d=lambda i,j:float(np.linalg.norm(p[i]-p[j]))
    return np.array([d(62,66)/max(d(60,64),1),
                     (d(37,41)+d(38,40))/(2*max(d(36,39),1)),
                     (d(43,47)+d(44,46))/(2*max(d(42,45),1))])


def diagnostics(source,target,result,same_input=False):
    identity=cosine(source.embedding,result.embedding)
    target_identity=cosine(target.embedding,result.embedding)
    pose=float(np.mean(np.abs(result.pose-target.pose)))
    differences=np.abs(aperture(result)-aperture(target))
    gates={'identity_cosine_at_least_0.35':identity>=.35,
           'identity_margin_at_least_0.05':same_input or identity-target_identity>=.05,
           'pose_mae_at_most_5deg':pose<=5,
           'mouth_error_at_most_0.08':bool(differences[0]<=.08),
           'eye_error_at_most_0.10':bool(differences[1:].mean()<=.1)}
    return {'identity_cosine':identity,'target_identity_cosine':target_identity,
            'source_target_cosine':cosine(source.embedding,target.embedding),
            'identity_margin':identity-target_identity,'pose_mae_deg':pose,
            'mouth_ratio_error':float(differences[0]),'eye_ratio_error':float(differences[1:].mean()),
            'gates':gates,'project_gate_pass':bool(all(gates.values())),
            'interpretation':'Heuristic project gates; cosine is not probability. Margin is inapplicable to identical inputs.'}
