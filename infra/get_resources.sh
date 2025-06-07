kubectl get nodes -o json | jq --slurpfile pods <(kubectl get pods -A -o json) '
def extract_pod:
  {
    podName: .metadata.name,
    podNamespace: .metadata.namespace,
    podNode: .spec.nodeName,
    podStatus: .status.phase,
    startTime: .status.startTime,
    containers: (
      .spec.containers | map({
        name: .name,
        resources: {
          limits: .resources.limits,
          requests: .resources.requests
        }
      })
    )
  };

(
  .items | map(. as $node | {
    node: $node.metadata.name,
    allocatable: $node.status.allocatable,
    gpu: $node.metadata.labels["nvidia.com/gpu.product"],
    pods: (
      $pods[0].items
      | map(extract_pod)
      | map(select(.podNode == $node.metadata.name))
    )
  })
  + [
    {
      node: "unscheduled",
      allocatable: null,
      gpu: null,
      pods: (
        $pods[0].items
        | map(extract_pod)
        | map(select(.podNode == null))
      )
    }
  ]
)' > result.json
