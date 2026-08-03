import torch
import torch.nn as nn
from torchvision import models

from src.codification import amplitude_encoding, angle_encoding

try:
    import pennylane as qml
except ImportError:
    qml = None


def _build_mlp(input_dim, hidden_layers, dropout, num_classes):
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)


def build_backbone(config):
    backbone_name = config.get("backbone", "resnet18").lower()
    pretrained = config.get("pretrained", True)
    freeze_backbone = config.get("freeze_backbone", True)

    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    return backbone, feature_dim


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.network = _build_mlp(
            input_dim,
            config.get("hidden_layers", [256]),
            config.get("dropout", 0.3),
            config["num_classes"],
        )

    def forward(self, features):
        return self.network(features)


class VariationalQuantumClassifier(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        if qml is None:
            raise ImportError("pennylane is required to use the quantum classifier.")

        quantum_config = config.get("quantum", {})
        self.encoding = quantum_config.get("encoding", "angle").lower()
        self.num_qubits = quantum_config.get("num_qubits", 4)
        self.num_layers = quantum_config.get("num_layers", 2)
        self.entanglement = quantum_config.get("entanglement", "linear").lower()
        self.input_dim = input_dim

        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            if self.encoding == "angle":
                qml.AngleEmbedding(
                    inputs,
                    wires=range(self.num_qubits),
                    rotation="Y",
                )
            elif self.encoding == "amplitude":
                qml.AmplitudeEmbedding(
                    inputs,
                    wires=range(self.num_qubits),
                    normalize=False,
                    pad_with=0.0,
                )
            else:
                raise ValueError(f"Unsupported quantum encoding: {self.encoding}")

            for layer in range(self.num_layers):
                for wire in range(self.num_qubits):
                    qml.RX(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                    qml.RZ(weights[layer, wire, 2], wires=wire)

                for wire in range(self.num_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                if self.entanglement == "circular" and self.num_qubits > 2:
                    qml.CNOT(wires=[self.num_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.num_qubits)]

        weight_shapes = {"weights": (self.num_layers, self.num_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.output_layer = nn.Linear(self.num_qubits, config["num_classes"])

    def _prepare_inputs(self, features):
        if self.encoding == "angle":
            return angle_encoding(features, self.num_qubits)
        if self.encoding == "amplitude":
            return amplitude_encoding(features, self.num_qubits)
        raise ValueError(f"Unsupported quantum encoding: {self.encoding}")

    def forward(self, features):
        encoded_features = self._prepare_inputs(features)

        try:
            quantum_outputs = self.quantum_layer(encoded_features)
            if quantum_outputs.ndim == 1:
                quantum_outputs = quantum_outputs.unsqueeze(0)
        except RuntimeError as error:
            if encoded_features.ndim == 1:
                raise error
            quantum_outputs = torch.stack(
                [self.quantum_layer(sample) for sample in encoded_features], dim=0
            )

        return self.output_layer(quantum_outputs.float())


def build_feature_classifier(input_dim, config):
    classifier_type = config.get("classifier", "mlp").lower()
    classifier_config = dict(config.get(classifier_type, {}))
    classifier_config["num_classes"] = config["num_classes"]

    if classifier_type == "mlp":
        return MLPClassifier(input_dim, classifier_config)
    if classifier_type == "quantum":
        return VariationalQuantumClassifier(input_dim, {**config, "quantum": classifier_config})

    raise ValueError(f"Unsupported classifier: {classifier_type}")


def build_model(config):
    backbone, feature_dim = build_backbone(config)
    classifier = MLPClassifier(feature_dim, {**config.get("mlp", {}), "num_classes": config["num_classes"]})
    return nn.Sequential(backbone, classifier)
